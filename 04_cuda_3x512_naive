#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define BATCH_SIZE 128
#define TRAIN_SIZE 50000
#define TEST_SIZE 10000
#define HIDDEN_SIZE 512
#define EPOCHS 40
#define LEARNING_RATE 0.01

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

typedef struct {
    double data_loading;
    double fwd_matmul1;
    double fwd_bias1;
    double fwd_relu;
    double fwd_matmul2;
    double fwd_bias2;
    double fwd_softmax;
    double cross_entropy;
    double bwd_output_grad;
    double bwd_matmul2;
    double bwd_bias2;
    double bwd_relu;
    double bwd_matmul1;
    double bwd_bias1;
    double weight_updates;
    double total_time;
} TimingStats;


typedef struct {
    float * weights1;
    float * weights2; 
    float * weights3;
    float * bias1;
    float * bias2;
    float * bias3;
    float * gradweights1;
    float * gradweights2;
    float * gradweights3;
    float * gradbias1;
    float * gradbias2;
    float * gradbias3;

}NeuralNetwork;

// load batched img data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}



__global__ void kernel_matmul_A_B_ (float* a, float* b , float* c , int M, int N, int K){// MKxKN=MN
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row<M&&col<N){
        float sum=0.0;
        for (int l=0;l<K;l++){
            sum+=a[K*row+l]*b[l*N+col];
        }
        c[row*N+col]=sum;
    }
}


__global__ void kernel_matmul_A_Bt_ (float* a, float* b , float* c , int M, int N, int K){// MKx(NK)t=MN
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row<M&&col<N){
        float sum=0.0;
        for (int l=0;l<K;l++){
            sum+=a[K*row+l]*b[K*col+l];
        }
        c[row*N+col]=sum;
    }
}


__global__ void kernel_matmul_At_B_ (float* a, float* b , float* c , int M, int N, int K){// (KM)txKN=MN
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row<M&&col<N){
        float sum=0.0;
        for (int l=0;l<K;l++){
            sum+=a[l*M+row]*b[l*N+col];
        }
        c[row*N+col]=sum;
    }
}

void init_weight(float* weights, int height, int width){// Perfoms he-initialization :
    float scale=sqrtf(2.0/height);
    for (int i = 0; i < height * width; i++) {
        weights[i]=((((float)rand() / RAND_MAX))*2.0-1.0)*scale;
    }
}

void init_bias(float* bias,int size){
    for (int i =0; i<size;i++){
        bias[i]=0.0;
    }
}
__global__ void kernel_relu(float * data ,int  size){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<size){
        if (data[idx]<0){
            data[idx]=0;
        }
    }
}
__global__ void kernel_bias_forward(float*data, float* bias, int size){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;

    int col=idx%size;
    int row=idx/size;
    if (row<BATCH_SIZE && col<size){
            data[idx]+=bias[col];
        }
    
}


__global__ void relu_back_kernel(float* x, float* dA1,int size ){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<size){
        dA1[idx]*=(x[idx]>0.0 ? 1.0:0.0);
    }
}

__global__ void kernel_bias_back(float *grad_bias, float *grad_output, int size) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<size){
        grad_bias[idx]=0.0;
        for (int b=0; b<BATCH_SIZE;b++)
            grad_bias[idx]+=grad_output[b*size+idx];
    }
}

__global__ void grad_0_kernel(float *grad, int size) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<size){
        grad[idx]=0;
    }
}
__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

void forward(NeuralNetwork* nn, float* batch_data, float* hidden, float* hidden2, float * output){
    dim3 block_size(32, 32);

    // W1 : batch_data (B x I) * weights1 (I x H) -> hidden (B x H)
    dim3 grid1_size((HIDDEN_SIZE+block_size.x-1)/block_size.x , ( BATCH_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_A_B_<<<grid1_size,block_size>>>(batch_data,nn->weights1, hidden , BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //b1
    kernel_bias_forward<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(hidden,nn->bias1,HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //Relu1
    kernel_relu<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, HIDDEN_SIZE*BATCH_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // W2 : hidden (B x H) * weights2 (H x H) -> hidden2 (B x H)
    dim3 grid2_size((HIDDEN_SIZE+block_size.x-1)/block_size.x , ( BATCH_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_A_B_<<<grid2_size,block_size>>>(hidden,nn->weights2, hidden2 , BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //b2
    kernel_bias_forward<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(hidden2,nn->bias2,HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //Relu2
    kernel_relu<<<(BATCH_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(hidden2, HIDDEN_SIZE*BATCH_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // W3 : hidden2 (B x H) * weights3 (H x O) -> output (B x O)
    dim3 grid3_size((OUTPUT_SIZE+block_size.x-1)/block_size.x , ( BATCH_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_A_B_<<<grid3_size,block_size>>>(hidden2,nn->weights3, output , BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //b3
    kernel_bias_forward<<<(BATCH_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(output,nn->bias3,OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //Softmax
    softmax_kernel<<<BATCH_SIZE, 1>>>(output, BATCH_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
}
float crossentropyloss(float* output, int * labels, int size){
    float total_loss=0.0;
    for (int b =0; b<BATCH_SIZE; b++){
        total_loss-=logf(fmaxf(output[size*b+labels[b]],1e-7));
    }
    return total_loss/BATCH_SIZE;
}
__global__ void kernel_crossentropyloss(float* d_output, int * d_labels, int size, float* batch_loss){
    __shared__ float s_loss[256];
    int tid = threadIdx.x;
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    s_loss[tid]=0.0;
    if (idx<BATCH_SIZE){
        s_loss[tid]=-logf(fmaxf(d_output[size*idx+d_labels[idx]],1e-7));
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_loss[tid] += s_loss[tid + s];
        __syncthreads();
    }
    if (tid==0){
        atomicAdd(batch_loss,s_loss[0]/BATCH_SIZE );
    }
}



__global__ void kernel_compute_output_gradients(float* output,float * grad_output,  int* batch_labels ){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<BATCH_SIZE){
        for(int i=0; i<OUTPUT_SIZE;i++){
            grad_output[idx*OUTPUT_SIZE+i]=output[idx*OUTPUT_SIZE+i];}
        grad_output[idx*OUTPUT_SIZE+batch_labels[idx]]-=1.0;
    
        for (int i = 0; i < OUTPUT_SIZE; i++) {
        grad_output[idx*OUTPUT_SIZE+i] /= BATCH_SIZE;
        }
    }
}
__global__ void kernel_update_weights(float* weights,float* grad_weights, int size, float lr){
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if (idx<size){
        weights[idx]-=lr*grad_weights[idx];
    }
}
void update_weights(NeuralNetwork * nn, float lr){
    
    kernel_update_weights<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->weights1,nn->gradweights1, INPUT_SIZE*HIDDEN_SIZE, lr);
    kernel_update_weights<<<(HIDDEN_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->weights2,nn->gradweights2, HIDDEN_SIZE*HIDDEN_SIZE, lr);
    kernel_update_weights<<<(HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(nn->weights3,nn->gradweights3, HIDDEN_SIZE*OUTPUT_SIZE, lr);
    kernel_update_weights<<<( HIDDEN_SIZE + 255) / 256, 256>>>(nn-> bias1,nn->gradbias1, HIDDEN_SIZE, lr);
    kernel_update_weights<<<( HIDDEN_SIZE + 255) / 256, 256>>>(nn-> bias2,nn->gradbias2, HIDDEN_SIZE, lr);
    kernel_update_weights<<<( OUTPUT_SIZE + 255) / 256, 256>>>(nn-> bias3,nn->gradbias3, OUTPUT_SIZE, lr);

}

void backward(NeuralNetwork* nn, float *batch_data,float*  hidden, float* hidden2, float*  output,int* batch_labels, int batch_size, float* grad_output, float* dA1, float* dA2){
    // Initialize gradients to zero
    grad_0_kernel<<<(HIDDEN_SIZE*INPUT_SIZE+255)/256,256>>>(nn->gradweights1,HIDDEN_SIZE * INPUT_SIZE);
    grad_0_kernel<<<(HIDDEN_SIZE*HIDDEN_SIZE+255)/256,256>>>(nn->gradweights2, HIDDEN_SIZE * HIDDEN_SIZE);
    grad_0_kernel<<<(HIDDEN_SIZE*OUTPUT_SIZE+255)/256,256>>>(nn->gradweights3, HIDDEN_SIZE * OUTPUT_SIZE);
    grad_0_kernel<<<(HIDDEN_SIZE+255)/256,256>>>(nn->gradbias1,HIDDEN_SIZE );
    grad_0_kernel<<<(HIDDEN_SIZE+255)/256,256>>>(nn->gradbias2,HIDDEN_SIZE );
    grad_0_kernel<<<(OUTPUT_SIZE+255)/256,256>>>(nn->gradbias3,OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    kernel_compute_output_gradients<<<(BATCH_SIZE+255)/256,256>>>(output,grad_output, batch_labels);
    CUDA_CHECK(cudaGetLastError());

    dim3 block_size(16,16);

    // ===================== LAYER 3 =====================
    dim3 grid3((OUTPUT_SIZE+block_size.x-1)/block_size.x,(HIDDEN_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_At_B_<<<grid3,block_size>>>(hidden2, grad_output,nn->gradweights3,  HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE );
    CUDA_CHECK(cudaGetLastError());
    kernel_bias_back<<<(OUTPUT_SIZE+255)/256,256>>>(nn->gradbias3, grad_output, OUTPUT_SIZE);

    // dA2 = grad_output @ W3.T
    dim3 grid_da2((HIDDEN_SIZE+block_size.x-1)/block_size.x,(BATCH_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_A_Bt_<<<grid_da2,block_size>>>(grad_output,nn->weights3,dA2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE );
    CUDA_CHECK(cudaGetLastError());
    relu_back_kernel<<<(BATCH_SIZE * HIDDEN_SIZE+255)/256,256>>>(hidden2,dA2,BATCH_SIZE * HIDDEN_SIZE );
    CUDA_CHECK(cudaGetLastError());

    // ===================== LAYER 2 =====================
    dim3 grid2((HIDDEN_SIZE+block_size.x-1)/block_size.x,(HIDDEN_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_At_B_<<<grid2,block_size>>>(hidden, dA2,nn->gradweights2,  HIDDEN_SIZE, HIDDEN_SIZE, BATCH_SIZE );
    CUDA_CHECK(cudaGetLastError());
    kernel_bias_back<<<(HIDDEN_SIZE+255)/256,256>>>(nn->gradbias2, dA2, HIDDEN_SIZE);

    // dA1 = dA2 @ W2.T
    dim3 grid_da1((HIDDEN_SIZE+block_size.x-1)/block_size.x,(BATCH_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_A_Bt_<<<grid_da1,block_size>>>(dA2,nn->weights2,dA1, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE );
    CUDA_CHECK(cudaGetLastError());
    relu_back_kernel<<<(BATCH_SIZE * HIDDEN_SIZE+255)/256,256>>>(hidden,dA1,BATCH_SIZE * HIDDEN_SIZE );
    CUDA_CHECK(cudaGetLastError());

    // ===================== LAYER 1 =====================
    dim3 grid1((HIDDEN_SIZE+block_size.x-1)/block_size.x,(INPUT_SIZE+block_size.y-1)/block_size.y);
    kernel_matmul_At_B_<<<grid1,block_size>>>(batch_data, dA1, nn->gradweights1,INPUT_SIZE, HIDDEN_SIZE,BATCH_SIZE );
    CUDA_CHECK(cudaGetLastError());
    kernel_bias_back<<<(HIDDEN_SIZE+255)/256,256>>>(nn->gradbias1, dA1, HIDDEN_SIZE);
}

void train(NeuralNetwork * nn, float* data, int* labels){
    float *dev_hidden, *dev_hidden2, *dev_output, *dev_batch;
    int *dev_batch_labels;

    CUDA_CHECK(cudaMalloc(&dev_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_hidden2, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_batch_labels, BATCH_SIZE * sizeof(int)));

    float *grad_output;
    CUDA_CHECK(cudaMalloc(&grad_output,OUTPUT_SIZE*BATCH_SIZE*sizeof(float) ));
    float* dA1;
    CUDA_CHECK(cudaMalloc(&dA1,HIDDEN_SIZE*BATCH_SIZE*sizeof(float)));
    float* dA2;
    CUDA_CHECK(cudaMalloc(&dA2,HIDDEN_SIZE*BATCH_SIZE*sizeof(float)));
    float *d_batch_loss;
    CUDA_CHECK(cudaMalloc(&d_batch_loss, sizeof(float)));


    for (int i=0;i<EPOCHS;i++){
        float epoch_tot_loss=0.0;

        for (int b =0;b<TRAIN_SIZE/BATCH_SIZE;b++){

            cudaMemset(d_batch_loss, 0, sizeof(float));   // reset loss for this batch

            float h_batch_loss;

            float* batch =&data[BATCH_SIZE*b*INPUT_SIZE];
            int* batch_labels=&labels[b*BATCH_SIZE];

            CUDA_CHECK(cudaMemcpy(dev_batch,batch, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_batch_labels,batch_labels,BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            
            forward(nn,dev_batch, dev_hidden, dev_hidden2, dev_output);
            
            kernel_crossentropyloss<<<(BATCH_SIZE + 255) / 256,256>>>(dev_output, dev_batch_labels, OUTPUT_SIZE,d_batch_loss);
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaMemcpy(&h_batch_loss,d_batch_loss,sizeof(float),cudaMemcpyDeviceToHost));
            epoch_tot_loss+=h_batch_loss;

            backward(nn, dev_batch, dev_hidden, dev_hidden2, dev_output, dev_batch_labels, BATCH_SIZE, grad_output,dA1,dA2);

            update_weights(nn, LEARNING_RATE);
        }
        printf("Epoch %d loss: %.4f\n", i, epoch_tot_loss / (TRAIN_SIZE/BATCH_SIZE));

    CUDA_CHECK(cudaDeviceSynchronize());

    }

    // free everything
    CUDA_CHECK(cudaFree(dev_hidden));
    CUDA_CHECK(cudaFree(dev_hidden2));
    CUDA_CHECK(cudaFree(dev_output));
    CUDA_CHECK(cudaFree(dev_batch));
    CUDA_CHECK(cudaFree(dev_batch_labels));
    CUDA_CHECK(cudaFree(grad_output));
    CUDA_CHECK(cudaFree(dA1));
    CUDA_CHECK(cudaFree(dA2));
    CUDA_CHECK(cudaFree(d_batch_loss));
}
void normalize_data(float* data,int size){
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i =0; i<size; i++){
        data[i]=(data[i]-mean)/std;
    }
}
void init_rand_weigths(NeuralNetwork * nn){
    // Create host buffers (3 layers)
    float* w1_h=(float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *w2_h = (float *)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    float *w3_h = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *b1_h = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *b2_h = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *b3_h = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize on host (He init)
    init_weight(w1_h,INPUT_SIZE, HIDDEN_SIZE);
    init_weight(w2_h, HIDDEN_SIZE, HIDDEN_SIZE);
    init_weight(w3_h, HIDDEN_SIZE, OUTPUT_SIZE);
    init_bias(b1_h, HIDDEN_SIZE);
    init_bias(b2_h, HIDDEN_SIZE);
    init_bias(b3_h, OUTPUT_SIZE);

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(nn->weights1, w1_h,INPUT_SIZE * HIDDEN_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, w2_h, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights3, w3_h, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, b1_h, HIDDEN_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, b2_h, HIDDEN_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias3, b3_h, OUTPUT_SIZE * sizeof(float),cudaMemcpyHostToDevice));
    
    // Free host buffers
    free(w1_h);
    free(w2_h);
    free(w3_h);
    free(b1_h);
    free(b2_h);
    free(b3_h);
}

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void init_NeuralNetwork(NeuralNetwork * nn){

    CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights3, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1,  HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias3, OUTPUT_SIZE* sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradweights3, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->gradbias3, OUTPUT_SIZE * sizeof(float)));

    init_rand_weigths( nn);
}
void evaluate(NeuralNetwork* nn){

}

int main(){
    srand(42);
    NeuralNetwork nn;

    init_NeuralNetwork(&nn);
    float * x_train;
    int * y_train;
    cudaMallocHost(&x_train, INPUT_SIZE*TRAIN_SIZE*sizeof(float));
    cudaMallocHost(&y_train, TRAIN_SIZE*sizeof(int));
    float * x_test=(float*)malloc(INPUT_SIZE*TEST_SIZE*sizeof(float));
    int * y_test=(int*)malloc(TEST_SIZE*sizeof(int));

    load_data("data/X_train.bin", x_train,INPUT_SIZE*TRAIN_SIZE );
    load_data("data/X_test.bin", x_test,INPUT_SIZE*TEST_SIZE );
    normalize_data(x_train, INPUT_SIZE*TRAIN_SIZE);
    normalize_data(x_test, INPUT_SIZE*TEST_SIZE);
    load_labels("data/y_train.bin", y_train,TRAIN_SIZE );
    load_labels("data/y_test.bin", y_test,TEST_SIZE );

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    train(&nn, x_train,y_train);
    clock_gettime(CLOCK_MONOTONIC, &end);
    float total_time = get_time_diff(start, end);
    printf("Total time is ");
    printf("  GPU compute:    %6.3fs\n", total_time);

    //free
    CUDA_CHECK(cudaFree(nn.weights1));
    CUDA_CHECK(cudaFree(nn.weights2));
    CUDA_CHECK(cudaFree(nn.weights3));
    CUDA_CHECK(cudaFree(nn.bias1));
    CUDA_CHECK(cudaFree(nn.bias2));
    CUDA_CHECK(cudaFree(nn.bias3));
    CUDA_CHECK(cudaFree(nn.gradweights1));
    CUDA_CHECK(cudaFree(nn.gradweights2));
    CUDA_CHECK(cudaFree(nn.gradweights3));
    CUDA_CHECK(cudaFree(nn.gradbias1));
    CUDA_CHECK(cudaFree(nn.gradbias2));
    CUDA_CHECK(cudaFree(nn.gradbias3));

    cudaFree(x_train);
    cudaFree(y_train);
    free(x_test);
    free(y_test);
    return 0;
}