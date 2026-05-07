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
#define TILE_SIZE     16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ─── Struct ──────────────────────────────────────────────────────────────────
// Master weights : FP32 (backward inchangé)
// Shadow  weights: BF16 (forward GEMMs via tensor cores)
typedef struct {
    // ── FP32 master (backward + SGD update) ──────────────────────────────────
    float *weights1;      // H × I  col-major
    float *weights2;      // O × H
    float *bias1;         // H  (biases gardés en FP32, petits, kernels custom)
    float *bias2;         // O
    float *gradweights1;
    float *gradweights2;
    float *gradbias1;
    float *gradbias2;

    // ── BF16 shadow (forward seulement, mis à jour après chaque update) ──────
    __nv_bfloat16 *weights1_bf16;   // H × I
    __nv_bfloat16 *weights2_bf16;   // O × H
} NeuralNetwork;

// ─── Data loading ─────────────────────────────────────────────────────────────
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening file: %s\n", filename); exit(1); }
    size_t n = fread(data, sizeof(float), size, file);
    if ((int)n != size) { fprintf(stderr, "Read error %s\n", filename); exit(1); }
    fclose(file);
}
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening file: %s\n", filename); exit(1); }
    size_t n = fread(labels, sizeof(int), size, file);
    if ((int)n != size) { fprintf(stderr, "Read error %s\n", filename); exit(1); }
    fclose(file);
}

// ─── Host init ────────────────────────────────────────────────────────────────
void init_weight(float *w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++)
        w[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
}
void init_bias(float *b, int size) {
    for (int i = 0; i < size; i++) b[i] = 0.0f;
}
void normalize_data(float *data, int size) {
    const float mean = 0.1307f, std = 0.3081f;
    for (int i = 0; i < size; i++) data[i] = (data[i] - mean) / std;
}

// ─── Cast kernel FP32 → BF16 ──────────────────────────────────────────────────
// Utilisé pour : poids shadow, batch input, hidden après bias+relu
__global__ void fp32_to_bf16_kernel(const float *src, __nv_bfloat16 *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ─── Sync FP32 master → BF16 shadow (après chaque update_weights) ────────────
void sync_weights_to_bf16(NeuralNetwork *nn) {
    int n1 = INPUT_SIZE  * HIDDEN_SIZE;
    int n2 = HIDDEN_SIZE * OUTPUT_SIZE;
    fp32_to_bf16_kernel<<<(n1 + 255) / 256, 256>>>(nn->weights1, nn->weights1_bf16, n1);
    fp32_to_bf16_kernel<<<(n2 + 255) / 256, 256>>>(nn->weights2, nn->weights2_bf16, n2);
    CUDA_CHECK(cudaGetLastError());
}

// ─── Kernels (inchangés) ──────────────────────────────────────────────────────
__global__ void kernel_bias_relu_forward(float *data, float *bias, int hidden_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int col = idx % hidden_size;
    int row = idx / hidden_size;
    if (row < BATCH_SIZE && col < hidden_size) {
        float v = data[idx] + bias[col];
        data[idx] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void kernel_bias_forward(float *data, float *bias, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int col = idx % size;
    int row = idx / size;
    if (row < BATCH_SIZE && col < size)
        data[idx] += bias[col];
}

__global__ void relu_back_kernel(float *post_relu, float *dA, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
        dA[idx] *= (post_relu[idx] > 0.0f ? 1.0f : 0.0f);
}

__global__ void softmax_kernel_warp(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    int idx = threadIdx.x;
    float *row = x + b * size;
    float val = (idx < size) ? row[idx] : -1e38f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    float max_val = __shfl_sync(0xffffffff, val, 0);
    float e = (idx < size) ? expf(row[idx] - max_val) : 0.0f;
    float sum = e;
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    float total = __shfl_sync(0xffffffff, sum, 0);
    if (idx < size)
        row[idx] = fmaxf(e / total, 1e-7f);
}

__global__ void kernel_crossentropyloss(float *d_output, int *d_labels,
                                        int size, float *batch_loss, float *epoch_loss) {
    __shared__ float s_loss[256];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    s_loss[tid] = 0.0f;
    if (idx < BATCH_SIZE)
        s_loss[tid] = -logf(fmaxf(d_output[size * idx + d_labels[idx]], 1e-7f));
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_loss[tid] += s_loss[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float contrib = s_loss[0] / BATCH_SIZE;
        atomicAdd(batch_loss,  contrib);
        atomicAdd(epoch_loss, contrib);
    }
}

__global__ void kernel_compute_output_gradients(float *output, float *grad_output, int *batch_labels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < BATCH_SIZE) {
        for (int i = 0; i < OUTPUT_SIZE; i++)
            grad_output[idx * OUTPUT_SIZE + i] = output[idx * OUTPUT_SIZE + i];
        grad_output[idx * OUTPUT_SIZE + batch_labels[idx]] -= 1.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++)
            grad_output[idx * OUTPUT_SIZE + i] /= BATCH_SIZE;
    }
}

__global__ void update_all_weights(
    float *w1, float *gw1,
    float *w2, float *gw2,
    float *b1, float *gb1,
    float *b2, float *gb2,
    float lr)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < INPUT_SIZE  * HIDDEN_SIZE) w1[idx] -= lr * gw1[idx];
    if (idx < HIDDEN_SIZE * OUTPUT_SIZE) w2[idx] -= lr * gw2[idx];
    if (idx < HIDDEN_SIZE) b1[idx] -= lr * gb1[idx];
    if (idx < OUTPUT_SIZE) b2[idx] -= lr * gb2[idx];
}

__global__ void argmax_kernel(float *output, int *predictions, int output_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCH_SIZE) {
        int best = 0;
        float best_val = output[b * output_size];
        for (int i = 1; i < output_size; i++) {
            if (output[b * output_size + i] > best_val) {
                best = i;
                best_val = output[b * output_size + i];
            }
        }
        predictions[b] = best;
    }
}

__global__ void count_correct_kernel(int *predictions, int *labels, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE && predictions[idx] == labels[idx])
        atomicAdd(count, 1);
}

// ─── Forward BF16 ─────────────────────────────────────────────────────────────
// batch_data    : FP32 GPU ptr (données brutes)
// batch_bf16    : buffer BF16 pré-alloué BATCH_SIZE×INPUT_SIZE  (réutilisé)
// hidden_bf16   : buffer BF16 pré-alloué BATCH_SIZE×HIDDEN_SIZE (réutilisé)
// hidden        : buffer FP32 (sortie GEMM1 + bias+relu ; utilisé en backward)
// output        : buffer FP32 (sortie GEMM2 + bias + softmax)
//
// Pipeline :
//   batch_data(FP32) ──cast──> batch_bf16
//   GEMM1: W1_bf16(H×I) × batch_bf16(I×B) → hidden(H×B, FP32)   [tensor cores]
//   bias+relu sur hidden (FP32)
//   hidden(FP32) ──cast──> hidden_bf16
//   GEMM2: W2_bf16(O×H) × hidden_bf16(H×B) → output(O×B, FP32)  [tensor cores]
//   bias + softmax sur output (FP32)
void forward(NeuralNetwork *nn, cublasHandle_t handle,
             float *batch_data,
             __nv_bfloat16 *batch_bf16,
             __nv_bfloat16 *hidden_bf16,
             float *hidden,
             float *output)
{
    const float alpha = 1.0f, beta = 0.0f;

    // ── Cast batch input FP32 → BF16 ─────────────────────────────────────────
    int n_batch  = BATCH_SIZE  * INPUT_SIZE;
    fp32_to_bf16_kernel<<<(n_batch  + 255) / 256, 256>>>(batch_data, batch_bf16,  n_batch);
    CUDA_CHECK(cudaGetLastError());

    // ── GEMM1 : hidden(H×B, FP32) = W1_bf16(H×I) × batch_bf16(I×B) ──────────
    // CUBLAS_COMPUTE_32F : accumulation FP32, tensor cores BF16 activés
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE,
        &alpha,
        nn->weights1_bf16, CUDA_R_16BF, HIDDEN_SIZE,
        batch_bf16,        CUDA_R_16BF, INPUT_SIZE,
        &beta,
        hidden,            CUDA_R_32F,  HIDDEN_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // ── Bias + ReLU en FP32 (inchangé) ───────────────────────────────────────
    kernel_bias_relu_forward<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        hidden, nn->bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // ── Cast hidden post-relu FP32 → BF16 ────────────────────────────────────
    int n_hidden = BATCH_SIZE * HIDDEN_SIZE;
    fp32_to_bf16_kernel<<<(n_hidden + 255) / 256, 256>>>(hidden, hidden_bf16, n_hidden);
    CUDA_CHECK(cudaGetLastError());

    // ── GEMM2 : output(O×B, FP32) = W2_bf16(O×H) × hidden_bf16(H×B) ─────────
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        OUTPUT_SIZE, BATCH_SIZE, HIDDEN_SIZE,
        &alpha,
        nn->weights2_bf16, CUDA_R_16BF, OUTPUT_SIZE,
        hidden_bf16,       CUDA_R_16BF, HIDDEN_SIZE,
        &beta,
        output,            CUDA_R_32F,  OUTPUT_SIZE,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // ── Bias + Softmax en FP32 (inchangé) ────────────────────────────────────
    kernel_bias_forward<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(
        output, nn->bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    softmax_kernel_warp<<<BATCH_SIZE, 32>>>(output, BATCH_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
}

// ─── Backward FP32 (inchangé) ────────────────────────────────────────────────
// Les gradients s'accumulent en FP32 sur les masters.
// `hidden` est FP32 (post bias+relu) → relu_back correct.
void backward(NeuralNetwork *nn, cublasHandle_t handle,
              float *batch_data, float *hidden, float *output,
              int *batch_labels, float *grad_output, float *dA1, float *d_ones) {
    const float alpha = 1.0f, beta = 0.0f;

    kernel_compute_output_gradients<<<(BATCH_SIZE + 255) / 256, 256>>>(
        output, grad_output, batch_labels);

    // ── Couche 2 ──────────────────────────────────────────────────────────────
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        OUTPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE,
        &alpha, grad_output, OUTPUT_SIZE,
        hidden, HIDDEN_SIZE,
        &beta, nn->gradweights2, OUTPUT_SIZE));

    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
        OUTPUT_SIZE, BATCH_SIZE,
        &alpha, grad_output, OUTPUT_SIZE,
        d_ones, 1,
        &beta, nn->gradbias2, 1));

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_SIZE, BATCH_SIZE, OUTPUT_SIZE,
        &alpha, nn->weights2, OUTPUT_SIZE,     // ← FP32 master pour le backward
        grad_output, OUTPUT_SIZE,
        &beta, dA1, HIDDEN_SIZE));

    relu_back_kernel<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        hidden, dA1, BATCH_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // ── Couche 1 ──────────────────────────────────────────────────────────────
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN_SIZE, INPUT_SIZE, BATCH_SIZE,
        &alpha, dA1, HIDDEN_SIZE,
        batch_data, INPUT_SIZE,
        &beta, nn->gradweights1, HIDDEN_SIZE));

    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
        HIDDEN_SIZE, BATCH_SIZE,
        &alpha, dA1, HIDDEN_SIZE,
        d_ones, 1,
        &beta, nn->gradbias1, 1));
}

// ─── Update weights + sync shadow ─────────────────────────────────────────────
void update_weights(NeuralNetwork *nn, float lr) {
    update_all_weights<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(
        nn->weights1, nn->gradweights1,
        nn->weights2, nn->gradweights2,
        nn->bias1,    nn->gradbias1,
        nn->bias2,    nn->gradbias2,
        lr);
    // Toujours synchroniser les shadows BF16 après chaque mise à jour SGD
    sync_weights_to_bf16(nn);
}

// ─── Evaluate ─────────────────────────────────────────────────────────────────
void evaluate(NeuralNetwork *nn, cublasHandle_t handle,
              float *x_test_gpu, int *y_test_gpu) {
    int n_batches = TEST_SIZE / BATCH_SIZE;

    float *dev_hidden, *dev_output;
    __nv_bfloat16 *dev_batch_bf16, *dev_hidden_bf16;
    int   *dev_predictions, *d_correct;

    CUDA_CHECK(cudaMalloc(&dev_hidden,      BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output,      BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_batch_bf16,  BATCH_SIZE * INPUT_SIZE  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_hidden_bf16, BATCH_SIZE * HIDDEN_SIZE * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_predictions, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_correct,       sizeof(int)));
    CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));

    for (int b = 0; b < n_batches; b++) {
        float *x_batch = x_test_gpu + (size_t)b * BATCH_SIZE * INPUT_SIZE;
        int   *y_batch = y_test_gpu + b * BATCH_SIZE;

        forward(nn, handle, x_batch, dev_batch_bf16, dev_hidden_bf16, dev_hidden, dev_output);

        argmax_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
            dev_output, dev_predictions, OUTPUT_SIZE);
        CUDA_CHECK(cudaGetLastError());

        count_correct_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
            dev_predictions, y_batch, d_correct);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_correct;
    CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  acc: %.2f%%\n", 100.0f * h_correct / TEST_SIZE);

    CUDA_CHECK(cudaFree(dev_hidden));
    CUDA_CHECK(cudaFree(dev_output));
    CUDA_CHECK(cudaFree(dev_batch_bf16));
    CUDA_CHECK(cudaFree(dev_hidden_bf16));
    CUDA_CHECK(cudaFree(dev_predictions));
    CUDA_CHECK(cudaFree(d_correct));
}

// ─── Train ────────────────────────────────────────────────────────────────────
void train(NeuralNetwork *nn,
           float *data, int *labels,
           float *x_test, int *y_test) {
    // Upload training data
    float *dev_images; int *dev_labels;
    CUDA_CHECK(cudaMalloc(&dev_images, (size_t)TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_images, data,   (size_t)TRAIN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev_labels, TRAIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_labels, labels, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Upload test data
    float *dev_x_test; int *dev_y_test;
    CUDA_CHECK(cudaMalloc(&dev_x_test, (size_t)TEST_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_x_test, x_test, (size_t)TEST_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev_y_test, TEST_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_y_test, y_test, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Buffers FP32
    float *dev_hidden, *dev_output;
    float *grad_output, *dA1, *d_ones;
    float *d_batch_loss, *d_epoch_loss;

    CUDA_CHECK(cudaMalloc(&dev_hidden,   BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output,   BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_output,  BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA1,          BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_epoch_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones,       BATCH_SIZE * sizeof(float)));

    float ones_h[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) ones_h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_ones, ones_h, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Buffers BF16 temporaires (réutilisés à chaque batch)
    __nv_bfloat16 *dev_batch_bf16, *dev_hidden_bf16;
    CUDA_CHECK(cudaMalloc(&dev_batch_bf16,  BATCH_SIZE * INPUT_SIZE  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dev_hidden_bf16, BATCH_SIZE * HIDDEN_SIZE * sizeof(__nv_bfloat16)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        CUDA_CHECK(cudaMemset(d_epoch_loss, 0, sizeof(float)));

        for (int b = 0; b < TRAIN_SIZE / BATCH_SIZE; b++) {
            CUDA_CHECK(cudaMemset(d_batch_loss, 0, sizeof(float)));

            float *batch_data   = dev_images + (size_t)b * BATCH_SIZE * INPUT_SIZE;
            int   *batch_labels = dev_labels + b * BATCH_SIZE;

            forward(nn, handle, batch_data, dev_batch_bf16, dev_hidden_bf16, dev_hidden, dev_output);

            kernel_crossentropyloss<<<(BATCH_SIZE + 255) / 256, 256>>>(
                dev_output, batch_labels, OUTPUT_SIZE, d_batch_loss, d_epoch_loss);
            CUDA_CHECK(cudaGetLastError());

            // backward utilise batch_data FP32 et dev_hidden FP32 (inchangé)
            backward(nn, handle, batch_data, dev_hidden, dev_output,
                     batch_labels, grad_output, dA1, d_ones);

            // update_weights met à jour FP32 masters puis sync vers BF16 shadows
            update_weights(nn, LEARNING_RATE);
        }

        float h_epoch_loss;
        CUDA_CHECK(cudaMemcpy(&h_epoch_loss, d_epoch_loss, sizeof(float), cudaMemcpyDeviceToHost));
        printf("Epoch %2d  loss: %.4f", epoch, h_epoch_loss / (TRAIN_SIZE / BATCH_SIZE));
        evaluate(nn, handle, dev_x_test, dev_y_test);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dev_images));       CUDA_CHECK(cudaFree(dev_labels));
    CUDA_CHECK(cudaFree(dev_x_test));       CUDA_CHECK(cudaFree(dev_y_test));
    CUDA_CHECK(cudaFree(dev_hidden));       CUDA_CHECK(cudaFree(dev_output));
    CUDA_CHECK(cudaFree(grad_output));      CUDA_CHECK(cudaFree(dA1));
    CUDA_CHECK(cudaFree(d_batch_loss));     CUDA_CHECK(cudaFree(d_epoch_loss));
    CUDA_CHECK(cudaFree(d_ones));
    CUDA_CHECK(cudaFree(dev_batch_bf16));
    CUDA_CHECK(cudaFree(dev_hidden_bf16));
}

// ─── Network lifecycle ────────────────────────────────────────────────────────
void init_NeuralNetwork(NeuralNetwork *nn) {
    // ── FP32 masters ──────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&nn->weights1,     (size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2,     (size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1,        HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2,        OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights1, (size_t)INPUT_SIZE  * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights2, (size_t)HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias1,    HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias2,    OUTPUT_SIZE * sizeof(float)));

    // ── BF16 shadows ──────────────────────────────────────────────────────────
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
    CUDA_CHECK(cudaMemcpy(nn->bias1,    b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2,    b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(w1); free(w2); free(b1); free(b2);

    // Initialiser les shadows BF16 à partir des masters FP32
    sync_weights_to_bf16(nn);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void free_NeuralNetwork(NeuralNetwork *nn) {
    CUDA_CHECK(cudaFree(nn->weights1));     CUDA_CHECK(cudaFree(nn->weights2));
    CUDA_CHECK(cudaFree(nn->bias1));        CUDA_CHECK(cudaFree(nn->bias2));
    CUDA_CHECK(cudaFree(nn->gradweights1)); CUDA_CHECK(cudaFree(nn->gradweights2));
    CUDA_CHECK(cudaFree(nn->gradbias1));    CUDA_CHECK(cudaFree(nn->gradbias2));
    CUDA_CHECK(cudaFree(nn->weights1_bf16));
    CUDA_CHECK(cudaFree(nn->weights2_bf16));
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    train(&nn, x_train, y_train, x_test, y_test);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double total = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Total training time: %.3fs\n", total);

    free_NeuralNetwork(&nn);
    cudaFreeHost(x_train);
    cudaFreeHost(y_train);
    free(x_test);
    free(y_test);
    return 0;
}
