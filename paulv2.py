import numpy as np
import time


TRAIN_SIZE=50000
TEST_SIZE=10000
epochs=10
learning_rate=0.01



x_train_np=np.fromfile("data/X_train.bin",dtype=np.float32)
y_train = np.fromfile("data/y_train.bin", dtype=np.int32)[:TRAIN_SIZE]

x_test_np=np.fromfile("data/X_test.bin",dtype=np.float32)
y_test = np.fromfile("data/y_test.bin", dtype=np.int32)


x_train = x_train_np[:TRAIN_SIZE * 28 * 28].reshape(TRAIN_SIZE, 1, 28, 28)
x_test=x_test_np[:TEST_SIZE * 28 * 28].reshape(TEST_SIZE,1,28,28) 
mean,std=x_train.mean(),x_train.std()
x_train = (x_train - mean) / std
x_test= (x_test - mean) / std


def initialize_weights(in_size, out_size):
    scale = (2.0 / in_size) ** 0.5
    return (np.random.rand(in_size, out_size)* 2.0 - 1.0) * scale
def initialize_bias(out_size):
    return np.zeros((1, out_size))
def softmax(x):
    x-=np.max(x, axis=1, keepdims=True)
    out=np.exp(x)
    return out/ np.sum(out,axis=1,keepdims=True)

def cross_entropy_loss(y_predicted,y_true):
    batch_size=y_predicted.shape[0]
    y_pred=softmax(y_predicted)

    correct_ones=np.log(y_pred[np.arange(batch_size), y_true])
    loss=-np.sum(correct_ones)/batch_size
    return loss
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1=initialize_weights(input_size,hidden_size)
        self.b1=initialize_bias(hidden_size)
        self.W2=initialize_weights(hidden_size,output_size)
        self.b2=initialize_bias(output_size)


    def forward(self,X):
        batch_size=X.shape[0]
        X_flat = X.reshape(batch_size, -1)
        Z1=X_flat@self.W1+self.b1
        A1=np.maximum(Z1,0)
        Z2=A1@self.W2+self.b2
        return Z2, (X_flat,Z1,A1)

    def backward(self,cache,grad_output):
        (X_flat,Z1,A1)=cache
        dW2 = A1.T @grad_output# dW2 = X2.T @dLoss
        db2 = np.sum(grad_output, axis=0, keepdims=True)
        dA1 = grad_output @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)#dreluoutput=drelu(X1)
        dW1 = X_flat.T @ dZ1# dW1 = X1.T @dreluoutput
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return dW1,db1,dW2,db2
    def update_weights(self, grad_weights1, grad_bias1, grad_weights2, grad_bias2,lr):
        self.W1-=lr*grad_weights1
        self.b1-=lr*grad_bias1
        self.W2-=lr*grad_weights2
        self.b2-=lr*grad_bias2

def train_timed(model, X_train, y_train, batch_size, epochs, learning_rate):
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    for j in range (epochs):
        total_loss=0.0
        for i in range(0,len(x_train),batch_size):
            start=i
            end=i+batch_size
            data=x_train[start:end]  
            labels=y_train[start:end]
            # Forward
            logits,my_cache =model.forward(data)
            # Loss compute and conversions
            loss = cross_entropy_loss(logits,labels)
            total_loss+=loss
            softmax_probs=softmax(logits)
            y_true_one_hot = np.zeros_like(logits)
            y_true_one_hot[np.arange(len(labels)), labels] = 1
            grad_output = (softmax_probs - y_true_one_hot) /batch_size
            
            # Backward
            grad_weights1, grad_bias1, grad_weights2, grad_bias2 = model.backward(my_cache,grad_output)
            model.update_weights(grad_weights1, grad_bias1, grad_weights2, grad_bias2,lr=learning_rate)
        avg_loss=total_loss/(len(x_train) // batch_size)
        print( avg_loss)
def evaluate(model, x_test,y_test,TEST_SIZE ):
    total_loss=0.0
    for i in range(0,len(x_test),batch_size):
        start=i
        end=i+batch_size
        data=x_train[start:end]  
        labels=y_train[start:end]
        # Forward
        logits,my_cache =model.forward(data)
        # Loss compute and conversions
        loss = cross_entropy_loss(logits,labels)
        total_loss+=loss
    avg_loss=total_loss/(len(x_train) // batch_size)
    print(f"Average test loss is", avg_loss)

if __name__ == "__main__":

    print(x_train.dtype)
    # Inits
    batch_size=128
    epochs=10
    lr=0.01
    total_loss=0.0
    my_model=NeuralNetwork(784,256,10)

    start = time.time()
    train_timed(my_model, x_train,y_train, batch_size, epochs, lr)
    end = time.time()
    print(f"Train time is ", end-start)
    start = time.time()
    evaluate(my_model,x_test,y_test,TEST_SIZE ) 
    end = time.time()
    print(f"Test time is ", end-start)
