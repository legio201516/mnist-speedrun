#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define BATCH_SIZE 128
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
#define TEST_SIZE 10000
#define HIDDEN_SIZE 512
#define EPOCHS 10
#define LEARNING_RATE 0.001

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
void matmulAB(float *a, float*b, float* c, int M, int N, int K ){
    for (int i = 0;i<M;i++){
        for (int j = 0;j<N;j++){
            float sum=0.0;
            for (int k = 0;k<K;k++){
                sum+=a[K*i+k]*b[k*N+j];
            }
            c[i*N+j]=sum;
        }
    }
}
void matmulABt(float *a, float*b, float* c, int M, int N, int K ){
    for (int i = 0;i<M;i++){
        for (int j = 0;j<N;j++){
            float sum=0.0;
            for (int k = 0;k<K;k++){
                sum+=a[K*i+k]*b[j*K+k];
            }
            c[i*N+j]=sum;
        }
    }
}
void matmulAtB(float *a, float*b, float* c, int M, int N, int K ){
    for (int i = 0;i<M;i++){
        for (int j = 0;j<N;j++){
            float sum=0.0;
            for (int k = 0;k<K;k++){
                sum+=a[M*k+i]*b[k*N+j];
            }
            c[i*N+j]=sum;
        }
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

void relu(float * data ,int  size){
    for (int i =0; i<size;i++){
        if (data[i]<0){
            data[i]=0;
        }
    }
}
void bias_forward(float*data, float* bias, int size){
    for (int i=0;i <BATCH_SIZE;i++){
        for (int k=0; k<size;k++){
            data[i*size+k]+=bias[k];
        }
    }
}

void bias_backward(float* grad_bias,float* grad_output, int size){
    for (int i =0;i<size; i++){
        grad_bias[i]=0.0;
        for (int b=0; b<BATCH_SIZE;b++)
            grad_bias[i]+=grad_output[b*size+i];
    }
}
void zero_grad(float* gradvec,int total_size){
    for (int i =0;i<total_size;i++){
        gradvec[i]=0;
    }

}
void softmax(float * output, int batch_size, int size){
    for (int b=0;b<batch_size;b++){
        float max=output[0];
        for (int i=0;i<size;i++){   
            if (output[b*size+i]>max){
                max=output[b*size+i];
            }
        }
        float sum=0.0;
        for (int i=0;i<size;i++){   
            output[b*size+i]=expf(output[b*size+i]-max);
            sum+=output[b*size+i];
        }
        for (int i = 0; i < size; i++) {
            output[b * size + i] = fmaxf(output[b * size + i] / sum, 1e-7f);
        }
    }   
}

void forward(NeuralNetwork* nn, float* batch_data, float* hidden, float* hidden2, float *output){
    matmulAB(batch_data,nn->weights1, hidden , BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE);
    bias_forward(hidden,nn->bias1,HIDDEN_SIZE);
    relu(hidden, HIDDEN_SIZE*BATCH_SIZE);
    matmulAB(hidden,nn->weights2, hidden2 , BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE);
    bias_forward(hidden2,nn->bias2,HIDDEN_SIZE);
    relu(hidden2, HIDDEN_SIZE*BATCH_SIZE);
    matmulAB(hidden2,nn->weights3, output , BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
    bias_forward(output,nn->bias3,OUTPUT_SIZE);
    softmax(output,BATCH_SIZE, OUTPUT_SIZE);

}
float crossentropyloss(float* output, int * labels, int size){
    float total_loss=0.0;
    for (int b =0; b<BATCH_SIZE; b++){
        total_loss-=logf(fmaxf(output[size*b+labels[b]],1e-7));
    }
    return total_loss/BATCH_SIZE;
}
float* compute_output_grad(float* output,float * grad_output,  int* batch_labels,int size ){
    for (int b =0; b<BATCH_SIZE; b++){
        for(int i=0; i<size;i++){
            grad_output[b*size+i]=output[b*size+i];}
        
        grad_output[b*size+batch_labels[b]]-=1;}
    for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; i++) {
        grad_output[i] /= BATCH_SIZE;
    }
    return grad_output;
}
void update_weights(NeuralNetwork * nn, float lr){
    for (int i =0; i<INPUT_SIZE*HIDDEN_SIZE;i++){
        nn->weights1[i]-=lr* nn->gradweights1[i];
    }
    for (int i =0; i<HIDDEN_SIZE*HIDDEN_SIZE;i++){
        nn->weights2[i]-=lr* nn->gradweights2[i];
    }
    for (int i =0; i<HIDDEN_SIZE;i++){
        nn->bias1[i]-=lr* nn->gradbias1[i];
    }
    for (int i =0; i<HIDDEN_SIZE;i++){
        nn->bias2[i]-=lr* nn->gradbias2[i];
    }
    for (int i =0; i<HIDDEN_SIZE*OUTPUT_SIZE;i++){
        nn->weights3[i]-=lr* nn->gradweights3[i];
    }
    for (int i =0; i<OUTPUT_SIZE;i++){
        nn->bias3[i]-=lr* nn->gradbias3[i];
    }
}

 
void backward(NeuralNetwork* nn, float *batch_data,float* hidden, float* hidden2, float* output,int* batch_labels){
/*
       (X_flat,Z1,A1,Z2,A2)=cache
        dW3 = A2.T @grad_output# dW3 = X2.T @dLoss
        db3 = np.sum(grad_output, axis=0, keepdims=True)
        dA2 = grad_output @ self.W3.T

        dZ2 = dA2 * (Z2 > 0)#dreluoutput=drelu(X2)
        dW2 = A1.T @dZ2# dW2 = X3.T @dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)#dreluoutput=drelu(X1)
        dW1 = X_flat.T @ dZ1# dW1 = X1.T @dreluoutput
        db1 = np.sum(dZ1, axis=0, keepdims=True)
*/
    zero_grad(nn->gradweights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->gradweights2, HIDDEN_SIZE * HIDDEN_SIZE);
    zero_grad(nn->gradweights3, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->gradbias1, HIDDEN_SIZE);
    zero_grad(nn->gradbias2, HIDDEN_SIZE);
    zero_grad(nn->gradbias3, OUTPUT_SIZE);

    float * grad_output=(float*) malloc(OUTPUT_SIZE*BATCH_SIZE*sizeof(float));
    compute_output_grad(output,grad_output,batch_labels,OUTPUT_SIZE );
    
    matmulAtB(hidden2, grad_output,nn->gradweights3,  HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE );
    bias_backward(nn->gradbias3, grad_output, OUTPUT_SIZE);
    float* dA2=(float*)malloc(HIDDEN_SIZE*BATCH_SIZE*sizeof(float));
    
    matmulABt(grad_output,nn->weights3,dA2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE );
    for (int i =0; i<BATCH_SIZE*HIDDEN_SIZE; i++){
        if (hidden2[i]<0){
            dA2[i]=0.0;
        }
    }
    matmulAtB(hidden, dA2,nn->gradweights2,  HIDDEN_SIZE, HIDDEN_SIZE, BATCH_SIZE );
    bias_backward(nn->gradbias2, dA2, HIDDEN_SIZE);
    float* dA1=(float*)malloc(HIDDEN_SIZE*BATCH_SIZE*sizeof(float));

    matmulABt(dA2,nn->weights2,dA1, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE );
    for (int i =0; i<BATCH_SIZE*HIDDEN_SIZE; i++){
        if (hidden[i]<0){
            dA2[i]=0.0;
        }
    }
    matmulAtB(batch_data, dA1, nn->gradweights1,INPUT_SIZE, HIDDEN_SIZE,BATCH_SIZE );
    bias_backward(nn->gradbias1, dA1, HIDDEN_SIZE);

    free(dA1);
    free(dA2);
    free(grad_output);
}
void train(NeuralNetwork * nn, float* data, int* labels){
    float * hidden=(float*)malloc(BATCH_SIZE*HIDDEN_SIZE*sizeof(float));
    float * hidden2=(float*)malloc(BATCH_SIZE*HIDDEN_SIZE*sizeof(float));
    float * output=(float*)malloc(BATCH_SIZE*OUTPUT_SIZE*sizeof(float));
    for (int i=0;i<EPOCHS;i++){
        float epoch_tot_loss=0.0;
        for (int b =0;b<TRAIN_SIZE/BATCH_SIZE;b++){
            float* batch =&data[BATCH_SIZE*b*INPUT_SIZE];
            int* batch_labels=&labels[b*BATCH_SIZE];
            forward(nn,batch, hidden,hidden2, output);
            epoch_tot_loss+=crossentropyloss(output,batch_labels, OUTPUT_SIZE);

            backward(nn, batch, hidden, hidden, output, batch_labels);

            update_weights(nn, LEARNING_RATE);
        }
        printf("Epoch %d loss: %.4f\n", i, epoch_tot_loss / (TRAIN_SIZE/BATCH_SIZE));


    }
    free(hidden2);
    free(hidden);
    free(output);

}
void normalize_data(float* data,int size){
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i =0; i<size; i++){
        data[i]=(data[i]-mean)/std;
    }
}

void init_NeuralNetwork(NeuralNetwork * nn){
    nn->weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weights2 = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weights3 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));

    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias3 = malloc(OUTPUT_SIZE * sizeof(float));
    
    nn->gradweights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->gradweights2 = malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->gradweights3 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
 
    nn->gradbias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->gradbias2 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->gradbias3 = malloc(OUTPUT_SIZE * sizeof(float));

    init_weight(nn->weights1,INPUT_SIZE,HIDDEN_SIZE );
    init_bias(nn->bias1, HIDDEN_SIZE);
    init_weight(nn->weights2,HIDDEN_SIZE,HIDDEN_SIZE );
    init_bias(nn->bias2, HIDDEN_SIZE);
    init_weight(nn->weights3,HIDDEN_SIZE,OUTPUT_SIZE );
    init_bias(nn->bias3, OUTPUT_SIZE);
}

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}
int main(){
    srand(time(NULL));
    NeuralNetwork nn;
 

    init_NeuralNetwork(&nn);
    //allocs
    float * x_train=(float*)malloc(INPUT_SIZE*TRAIN_SIZE*sizeof(float));
    int * y_train=(int*)malloc(TRAIN_SIZE*sizeof(int));
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
    printf("  GPU compute:    %6.3fs ", total_time);
    //free

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(nn.weights3);
    free(nn.bias3);

    free(nn.gradweights1);
    free(nn.gradweights2);
    free(nn.gradweights3);
    free(nn.gradbias3);
    free(nn.gradbias1);
    free(nn.gradbias2);
    free(x_train);
    free(y_train);
    free(x_test);
    free(y_test);
    return 0;
}