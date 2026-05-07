#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#define INPUT_SIZE    784
#define OUTPUT_SIZE   10
#define BATCH_SIZE    1024
#define TRAIN_SIZE    50000
#define TEST_SIZE     10000
#define HIDDEN_SIZE   512
#define EPOCHS        40
#define LEARNING_RATE 0.01f

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ─── Struct ───────────────────────────────────────────────────────────────────
//
//  FP32 masters  : source de vérité pour SGD + accumulateurs de gradients
//  BF16 shadows  : projetés depuis les masters avant chaque forward/backward GEMM
//
typedef struct {
    // FP32 masters
    float *weights1;       // H×I col-major
    float *weights2;       // O×H
    float *bias1;          // H  (gardé FP32 : vecteur petit, pas de gemv BF16)
    float *bias2;          // O
    float *gradweights1;   // H×I  FP32 accumulateur
    float *gradweights2;   // O×H
    float *gradbias1;      // H
    float *gradbias2;      // O

    // BF16 shadows (forward + backward GEMMs)
    __nv_bfloat16 *weights1_bf16;   // H×I
    __nv_bfloat16 *weights2_bf16;   // O×H
} NeuralNetwork;

// ─── Data I/O ─────────────────────────────────────────────────────────────────
void load_data(const char *fn, float *data, int n) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Error opening %s\n", fn); exit(1); }
    if ((int)fread(data, sizeof(float), n, f) != n) { fprintf(stderr, "Read error %s\n", fn); exit(1); }
    fclose(f);
}
void load_labels(const char *fn, int *labels, int n) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Error opening %s\n", fn); exit(1); }
    if ((int)fread(labels, sizeof(int), n, f) != n) { fprintf(stderr, "Read error %s\n", fn); exit(1); }
    fclose(f);
}

// ─── Host helpers ─────────────────────────────────────────────────────────────
void init_weight(float *w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++)
        w[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
}
void init_bias(float *b, int n)  { for (int i = 0; i < n; i++) b[i] = 0.0f; }
void normalize_data(float *d, int n) {
    const float mean = 0.1307f, std = 0.3081f;
    for (int i = 0; i < n; i++) d[i] = (d[i] - mean) / std;
}

// ─── Cast FP32 → BF16 ─────────────────────────────────────────────────────────
__global__ void fp32_to_bf16_kernel(const float *src, __nv_bfloat16 *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
static inline void cast_f32_to_bf16(const float *src, __nv_bfloat16 *dst, int n) {
    fp32_to_bf16_kernel<<<(n + 255) / 256, 256>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

// Sync FP32 masters → BF16 shadows  (après init et après chaque update_weights)
void sync_weights_to_bf16(NeuralNetwork *nn) {
    cast_f32_to_bf16(nn->weights1, nn->weights1_bf16, INPUT_SIZE  * HIDDEN_SIZE);
    cast_f32_to_bf16(nn->weights2, nn->weights2_bf16, HIDDEN_SIZE * OUTPUT_SIZE);
}

// ─── Kernels (inchangés) ──────────────────────────────────────────────────────
__global__ void kernel_bias_relu_forward(float *data, float *bias, int hs) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < BATCH_SIZE * hs) {
        float v = data[idx] + bias[idx % hs];
        data[idx] = v > 0.0f ? v : 0.0f;
    }
}
__global__ void kernel_bias_forward(float *data, float *bias, int sz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < BATCH_SIZE * sz)
        data[idx] += bias[idx % sz];
}
__global__ void relu_back_kernel(float *post_relu, float *dA, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) dA[idx] *= (post_relu[idx] > 0.0f ? 1.0f : 0.0f);
}
__global__ void softmax_kernel_warp(float *x, int bs, int sz) {
    int b = blockIdx.x;
    if (b >= bs) return;
    int idx = threadIdx.x;
    float *row = x + b * sz;
    float val = (idx < sz) ? row[idx] : -1e38f;
    for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    float mx = __shfl_sync(0xffffffff, val, 0);
    float e   = (idx < sz) ? expf(row[idx] - mx) : 0.0f;
    float sum = e;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    float tot = __shfl_sync(0xffffffff, sum, 0);
    if (idx < sz) row[idx] = fmaxf(e / tot, 1e-7f);
}
__global__ void kernel_crossentropyloss(float *out, int *labels, int sz,
                                        float *bloss, float *eloss) {
    __shared__ float s[256];
    int tid = threadIdx.x, idx = blockDim.x * blockIdx.x + tid;
    s[tid] = (idx < BATCH_SIZE) ? -logf(fmaxf(out[sz * idx + labels[idx]], 1e-7f)) : 0.0f;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    if (tid == 0) { float c = s[0] / BATCH_SIZE; atomicAdd(bloss, c); atomicAdd(eloss, c); }
}
__global__ void kernel_compute_output_gradients(float *out, float *grad, int *labels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < BATCH_SIZE) {
        for (int i = 0; i < OUTPUT_SIZE; i++)
            grad[idx * OUTPUT_SIZE + i] = out[idx * OUTPUT_SIZE + i];
        grad[idx * OUTPUT_SIZE + labels[idx]] -= 1.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++)
            grad[idx * OUTPUT_SIZE + i] /= BATCH_SIZE;
    }
}
__global__ void update_all_weights(float *w1, float *gw1, float *w2, float *gw2,
                                   float *b1, float *gb1, float *b2, float *gb2, float lr) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < INPUT_SIZE  * HIDDEN_SIZE) w1[idx] -= lr * gw1[idx];
    if (idx < HIDDEN_SIZE * OUTPUT_SIZE) w2[idx] -= lr * gw2[idx];
    if (idx < HIDDEN_SIZE) b1[idx] -= lr * gb1[idx];
    if (idx < OUTPUT_SIZE) b2[idx] -= lr * gb2[idx];
}
__global__ void argmax_kernel(float *out, int *preds, int sz) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCH_SIZE) {
        int best = 0; float bv = out[b * sz];
        for (int i = 1; i < sz; i++) if (out[b * sz + i] > bv) { best = i; bv = out[b * sz + i]; }
        preds[b] = best;
    }
}
__global__ void count_correct_kernel(int *preds, int *labels, int *cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE && preds[idx] == labels[idx]) atomicAdd(cnt, 1);
}

// ─── Forward BF16 ─────────────────────────────────────────────────────────────
//
//  Entrées FP32  → castées en BF16 pour les GEMMs (tensor cores Ada)
//  Sorties GEMM  → FP32 (CUBLAS_COMPUTE_32F : accumulation FP32)
//  Biais + ReLU  → FP32 custom kernels (inchangés)
//
//  batch_data(FP32) ──cast──► batch_bf16(BF16)
//  GEMM1 : W1_bf16(H×I) × batch_bf16(I×B) → hidden(FP32, H×B)
//  bias + ReLU sur hidden FP32
//  hidden(FP32) ──cast──► hidden_bf16(BF16)   ← réutilisé en backward
//  GEMM2 : W2_bf16(O×H) × hidden_bf16(H×B) → output(FP32, O×B)
//  bias + softmax sur output FP32
//
void forward(NeuralNetwork *nn, cublasHandle_t handle,
             float          *batch_data,
             __nv_bfloat16  *batch_bf16,    // [B×I] BF16 — rempli ici, relu en backward
             __nv_bfloat16  *hidden_bf16,   // [B×H] BF16 — rempli ici, relu en backward
             float          *hidden,        // [B×H] FP32 — relu mask + backward
             float          *output)        // [B×O] FP32
{
    const float alpha = 1.0f, beta = 0.0f;

    // Cast batch FP32 → BF16
    cast_f32_to_bf16(batch_data, batch_bf16, BATCH_SIZE * INPUT_SIZE);

    // GEMM1 : hidden(H×B, FP32) = W1_bf16(H×I) × batch_bf16(I×B)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE,
        &alpha,
        nn->weights1_bf16, CUDA_R_16BF, HIDDEN_SIZE,
        batch_bf16,        CUDA_R_16BF, INPUT_SIZE,
        &beta,
        hidden,            CUDA_R_32F,  HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Bias + ReLU FP32
    kernel_bias_relu_forward<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        hidden, nn->bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Cast hidden post-relu FP32 → BF16  (réutilisé dans backward pour gradweights2)
    cast_f32_to_bf16(hidden, hidden_bf16, BATCH_SIZE * HIDDEN_SIZE);

    // GEMM2 : output(O×B, FP32) = W2_bf16(O×H) × hidden_bf16(H×B)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        OUTPUT_SIZE, BATCH_SIZE, HIDDEN_SIZE,
        &alpha,
        nn->weights2_bf16, CUDA_R_16BF, OUTPUT_SIZE,
        hidden_bf16,       CUDA_R_16BF, HIDDEN_SIZE,
        &beta,
        output,            CUDA_R_32F,  OUTPUT_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Bias + Softmax FP32
    kernel_bias_forward<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(
        output, nn->bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    softmax_kernel_warp<<<BATCH_SIZE, 32>>>(output, BATCH_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
}

// ─── Backward BF16 ────────────────────────────────────────────────────────────
//
//  Toutes les GEMMs utilisent des opérandes BF16 avec accumulation FP32.
//  Les accumulateurs de gradients (gradweights*, gradbias*) restent FP32 :
//  c'est là que la précision compte pour SGD.
//
//  batch_bf16, hidden_bf16 : réutilisés depuis forward (valeurs toujours valides)
//
//  Pipeline :
//    kernel → grad_output(FP32) ──cast──► grad_output_bf16
//
//    GEMM  gradweights2(FP32, O×H) = grad_output_bf16(O×B) × hidden_bf16^T(B×H)
//    GEMV  gradbias2(FP32)         = grad_output(FP32, O×B) × ones(B)   [pas de gemv BF16]
//
//    GEMM  dA1(FP32, H×B)          = W2_bf16^T(H×O)         × grad_output_bf16(O×B)
//    relu_back sur dA1(FP32)  — mask relu = (hidden FP32 > 0)
//    dA1(FP32) ──cast──► dA1_bf16
//
//    GEMM  gradweights1(FP32, H×I) = dA1_bf16(H×B)           × batch_bf16^T(B×I)
//    GEMV  gradbias1(FP32)         = dA1(FP32, H×B)           × ones(B)
//
void backward(NeuralNetwork *nn, cublasHandle_t handle,
              float          *hidden,           // FP32 post-relu (mask relu)
              float          *output,           // FP32 softmax out
              int            *batch_labels,
              float          *grad_output,      // FP32 buffer [B×O]
              float          *dA1,              // FP32 buffer [B×H]
              float          *d_ones,           // FP32 ones [B]
              __nv_bfloat16  *batch_bf16,       // BF16 [B×I]  — depuis forward
              __nv_bfloat16  *hidden_bf16,      // BF16 [B×H]  — depuis forward
              __nv_bfloat16  *grad_output_bf16, // BF16 [B×O]  — rempli ici
              __nv_bfloat16  *dA1_bf16)         // BF16 [B×H]  — rempli ici
{
    const float alpha = 1.0f, beta = 0.0f;

    // ── grad_output = (softmax − onehot) / B  [FP32] ─────────────────────────
    kernel_compute_output_gradients<<<(BATCH_SIZE + 255) / 256, 256>>>(
        output, grad_output, batch_labels);
    CUDA_CHECK(cudaGetLastError());

    // Cast grad_output FP32 → BF16  (pour les GEMMs couche 2)
    cast_f32_to_bf16(grad_output, grad_output_bf16, BATCH_SIZE * OUTPUT_SIZE);

    // ── Couche 2 ──────────────────────────────────────────────────────────────

    // gradweights2(FP32, O×H) = grad_output_bf16(O×B) × hidden_bf16^T(B×H)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        OUTPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE,
        &alpha,
        grad_output_bf16, CUDA_R_16BF, OUTPUT_SIZE,
        hidden_bf16,      CUDA_R_16BF, HIDDEN_SIZE,
        &beta,
        nn->gradweights2, CUDA_R_32F,  OUTPUT_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // gradbias2(FP32) = grad_output(FP32) × ones(B)
    // cublasSgemv : pas de version BF16, vecteur O petit → FP32 ok
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
        OUTPUT_SIZE, BATCH_SIZE,
        &alpha, grad_output, OUTPUT_SIZE,
        d_ones, 1,
        &beta, nn->gradbias2, 1));

    // dA1(FP32, H×B) = W2_bf16^T(H×O) × grad_output_bf16(O×B)
    // Note : on utilise le shadow BF16 de W2 (cohérent avec forward)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_SIZE, BATCH_SIZE, OUTPUT_SIZE,
        &alpha,
        nn->weights2_bf16, CUDA_R_16BF, OUTPUT_SIZE,  // lda = O (non-transposé)
        grad_output_bf16,  CUDA_R_16BF, OUTPUT_SIZE,
        &beta,
        dA1,               CUDA_R_32F,  HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // ReLU backward : mask FP32 (hidden post-relu > 0 ⟺ pre-relu > 0)
    relu_back_kernel<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        hidden, dA1, BATCH_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Cast dA1 FP32 → BF16  (pour les GEMMs couche 1)
    cast_f32_to_bf16(dA1, dA1_bf16, BATCH_SIZE * HIDDEN_SIZE);

    // ── Couche 1 ──────────────────────────────────────────────────────────────

    // gradweights1(FP32, H×I) = dA1_bf16(H×B) × batch_bf16^T(B×I)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_SIZE, INPUT_SIZE, BATCH_SIZE,
        &alpha,
        dA1_bf16,    CUDA_R_16BF, HIDDEN_SIZE,
        batch_bf16,  CUDA_R_16BF, INPUT_SIZE,
        &beta,
        nn->gradweights1, CUDA_R_32F, HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // gradbias1(FP32) = dA1(FP32) × ones(B)
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
        HIDDEN_SIZE, BATCH_SIZE,
        &alpha, dA1, HIDDEN_SIZE,
        d_ones, 1,
        &beta, nn->gradbias1, 1));
}

// ─── Update + sync shadows ────────────────────────────────────────────────────
void update_weights(NeuralNetwork *nn, float lr) {
    update_all_weights<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        nn->weights1, nn->gradweights1,
        nn->weights2, nn->gradweights2,
        nn->bias1,    nn->gradbias1,
        nn->bias2,    nn->gradbias2,
        lr);
    CUDA_CHECK(cudaGetLastError());
    // Reprojeter les masters FP32 mis à jour vers les shadows BF16
    sync_weights_to_bf16(nn);
}

// ─── Evaluate ─────────────────────────────────────────────────────────────────
void evaluate(NeuralNetwork *nn, cublasHandle_t handle,
              float *x_test_gpu, int *y_test_gpu) {
    float *dev_hidden, *dev_output;
    __nv_bfloat16 *dev_batch_bf16, *dev_hidden_bf16;
    int *dev_preds, *d_correct;

    CUDA_CHECK(cudaMalloc(&dev_hidden,      BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output,      BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_batch_bf16,  BATCH_SIZE * INPUT_SIZE  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_hidden_bf16, BATCH_SIZE * HIDDEN_SIZE * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_preds,       BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_correct,       sizeof(int)));
    CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));

    for (int b = 0; b < TEST_SIZE / BATCH_SIZE; b++) {
        forward(nn, handle,
                x_test_gpu + (size_t)b * BATCH_SIZE * INPUT_SIZE,
                dev_batch_bf16, dev_hidden_bf16, dev_hidden, dev_output);

        argmax_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(dev_output, dev_preds, OUTPUT_SIZE);
        CUDA_CHECK(cudaGetLastError());
        count_correct_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
            dev_preds, y_test_gpu + b * BATCH_SIZE, d_correct);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_correct;
    CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  acc: %.2f%%\n", 100.0f * h_correct / TEST_SIZE);

    CUDA_CHECK(cudaFree(dev_hidden));   CUDA_CHECK(cudaFree(dev_output));
    CUDA_CHECK(cudaFree(dev_batch_bf16)); CUDA_CHECK(cudaFree(dev_hidden_bf16));
    CUDA_CHECK(cudaFree(dev_preds));    CUDA_CHECK(cudaFree(d_correct));
}

// ─── Train ────────────────────────────────────────────────────────────────────
void train(NeuralNetwork *nn, float *data, int *labels, float *x_test, int *y_test) {

    // Upload données sur GPU
    float *dev_images; int *dev_labels;
    CUDA_CHECK(cudaMalloc(&dev_images, (size_t)TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_images, data,   (size_t)TRAIN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev_labels, TRAIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_labels, labels, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    float *dev_x_test; int *dev_y_test;
    CUDA_CHECK(cudaMalloc(&dev_x_test, (size_t)TEST_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_x_test, x_test, (size_t)TEST_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev_y_test, TEST_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_y_test, y_test, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Buffers FP32
    float *dev_hidden, *dev_output, *grad_output, *dA1, *d_ones, *d_bloss, *d_eloss;
    CUDA_CHECK(cudaMalloc(&dev_hidden,  BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output,  BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA1,         BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bloss,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eloss,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones,      BATCH_SIZE * sizeof(float)));

    float ones_h[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) ones_h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_ones, ones_h, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Buffers BF16 — alloués une seule fois, réutilisés à chaque batch
    // batch_bf16 et hidden_bf16 sont produits en forward puis consommés en backward
    __nv_bfloat16 *dev_batch_bf16, *dev_hidden_bf16, *dev_grad_output_bf16, *dev_dA1_bf16;
    CUDA_CHECK(cudaMalloc(&dev_batch_bf16,        BATCH_SIZE * INPUT_SIZE  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_hidden_bf16,       BATCH_SIZE * HIDDEN_SIZE * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_grad_output_bf16,  BATCH_SIZE * OUTPUT_SIZE * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_dA1_bf16,          BATCH_SIZE * HIDDEN_SIZE * sizeof(__nv_bfloat16)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        CUDA_CHECK(cudaMemset(d_eloss, 0, sizeof(float)));

        for (int b = 0; b < TRAIN_SIZE / BATCH_SIZE; b++) {
            CUDA_CHECK(cudaMemset(d_bloss, 0, sizeof(float)));

            float *batch_data   = dev_images + (size_t)b * BATCH_SIZE * INPUT_SIZE;
            int   *batch_labels = dev_labels + b * BATCH_SIZE;

            // forward : produit batch_bf16, hidden_bf16, hidden(FP32), output(FP32)
            forward(nn, handle, batch_data,
                    dev_batch_bf16, dev_hidden_bf16, dev_hidden, dev_output);

            kernel_crossentropyloss<<<(BATCH_SIZE + 255) / 256, 256>>>(
                dev_output, batch_labels, OUTPUT_SIZE, d_bloss, d_eloss);
            CUDA_CHECK(cudaGetLastError());

            // backward : réutilise batch_bf16 + hidden_bf16 issus du forward
            backward(nn, handle,
                     dev_hidden, dev_output, batch_labels,
                     grad_output, dA1, d_ones,
                     dev_batch_bf16, dev_hidden_bf16,
                     dev_grad_output_bf16, dev_dA1_bf16);

            // update FP32 masters + resync BF16 shadows
            update_weights(nn, LEARNING_RATE);
        }

        float h_eloss;
        CUDA_CHECK(cudaMemcpy(&h_eloss, d_eloss, sizeof(float), cudaMemcpyDeviceToHost));
        printf("Epoch %2d  loss: %.4f", epoch, h_eloss / (TRAIN_SIZE / BATCH_SIZE));
        evaluate(nn, handle, dev_x_test, dev_y_test);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dev_images));  CUDA_CHECK(cudaFree(dev_labels));
    CUDA_CHECK(cudaFree(dev_x_test));  CUDA_CHECK(cudaFree(dev_y_test));
    CUDA_CHECK(cudaFree(dev_hidden));  CUDA_CHECK(cudaFree(dev_output));
    CUDA_CHECK(cudaFree(grad_output)); CUDA_CHECK(cudaFree(dA1));
    CUDA_CHECK(cudaFree(d_bloss));     CUDA_CHECK(cudaFree(d_eloss));
    CUDA_CHECK(cudaFree(d_ones));
    CUDA_CHECK(cudaFree(dev_batch_bf16));
    CUDA_CHECK(cudaFree(dev_hidden_bf16));
    CUDA_CHECK(cudaFree(dev_grad_output_bf16));
    CUDA_CHECK(cudaFree(dev_dA1_bf16));
}

// ─── Network lifecycle ────────────────────────────────────────────────────────
void init_NeuralNetwork(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1,     (size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2,     (size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1,        HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2,        OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights1, (size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights2, (size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias1,    HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias2,    OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights1_bf16, (size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&nn->weights2_bf16, (size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(__nv_bfloat16)));

    float *w1 = (float*)malloc((size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(float));
    float *w2 = (float*)malloc((size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    init_weight(w1, INPUT_SIZE,  HIDDEN_SIZE);
    init_weight(w2, HIDDEN_SIZE, OUTPUT_SIZE);
    init_bias(b1, HIDDEN_SIZE);
    init_bias(b2, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpy(nn->weights1, w1, (size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, w2, (size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(w1); free(w2); free(b1); free(b2);

    // Initialiser les shadows BF16 depuis les masters FP32
    sync_weights_to_bf16(nn);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void free_NeuralNetwork(NeuralNetwork *nn) {
    CUDA_CHECK(cudaFree(nn->weights1));       CUDA_CHECK(cudaFree(nn->weights2));
    CUDA_CHECK(cudaFree(nn->bias1));          CUDA_CHECK(cudaFree(nn->bias2));
    CUDA_CHECK(cudaFree(nn->gradweights1));   CUDA_CHECK(cudaFree(nn->gradweights2));
    CUDA_CHECK(cudaFree(nn->gradbias1));      CUDA_CHECK(cudaFree(nn->gradbias2));
    CUDA_CHECK(cudaFree(nn->weights1_bf16));  CUDA_CHECK(cudaFree(nn->weights2_bf16));
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    srand(42);

    NeuralNetwork nn;
    init_NeuralNetwork(&nn);

    float *x_train, *x_test;
    int   *y_train, *y_test;

    cudaMallocHost(&x_train, (size_t)TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMallocHost(&y_train, TRAIN_SIZE * sizeof(int));
    x_test = (float*)malloc((size_t)TEST_SIZE * INPUT_SIZE * sizeof(float));
    y_test = (int*)  malloc(TEST_SIZE * sizeof(int));

    load_data("data/X_train.bin", x_train, INPUT_SIZE * TRAIN_SIZE);
    load_data("data/X_test.bin",  x_test,  INPUT_SIZE * TEST_SIZE);
    normalize_data(x_train, INPUT_SIZE * TRAIN_SIZE);
    normalize_data(x_test,  INPUT_SIZE * TEST_SIZE);
    load_labels("data/y_train.bin", y_train, TRAIN_SIZE);
    load_labels("data/y_test.bin",  y_test,  TEST_SIZE);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    train(&nn, x_train, y_train, x_test, y_test);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("Total training time: %.3fs\n", total);

    free_NeuralNetwork(&nn);
    cudaFreeHost(x_train); cudaFreeHost(y_train);
    free(x_test); free(y_test);
    return 0;
}
