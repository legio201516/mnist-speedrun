import torch
import numpy as np 
import torch.nn as nn
import time
import torch.optim as optim

TRAIN_SIZE=50000
TEST_SIZE=10000
epochs=10
learning_rate=0.01
batch_size=128
one_run=TRAIN_SIZE//batch_size

x_train_np=np.fromfile("data/X_train.bin",dtype=np.float32)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)

x_test_np=np.fromfile("data/X_test.bin",dtype=np.float32)
y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)


X_train_np = x_train_np[:TRAIN_SIZE * 28 * 28].reshape(TRAIN_SIZE, 1, 28, 28)
X_test_np=x_test_np[:TEST_SIZE * 28 * 28].reshape(TEST_SIZE,1,28,28) 
mean,std=X_train_np.mean(),X_train_np.std()
X_train_np = (X_train_np - mean) / std
X_test_np= (X_test_np - mean) / std
# To cuda 

x_train=torch.from_numpy(X_train_np).to('cuda')
x_test=torch.from_numpy(X_test_np).to('cuda')
y_test = torch.from_numpy(y_test_np[:TEST_SIZE]).long().to("cuda")
y_train=torch.from_numpy(y_train_np).long().to('cuda')

class CUDATimer:
    def __init__(self, stats_dict, key):
        self.stats = stats_dict
        self.key = key
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        self.start.record()
        return self
    
    def __exit__(self, *args):
        self.end.record()
        self.end.synchronize()
        self.stats[self.key] += self.start.elapsed_time(self.end) / 1000

class MLP(nn.Module):
    def __init__ (self, in_features, hidden_features):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, 10))

    def forward(self, x):  
        return self.layers(x)

#Train model while writing timing stats
def training(yourmodel,yourdata,yourlabels,totalepochs, stats):

    myloss=nn.CrossEntropyLoss()
    myoptimizer=optim.SGD(yourmodel.parameters(), lr=learning_rate)
    for j in range(totalepochs):
        print(j)
        epoch_loss=0.0
        for i in range(one_run):
            with CUDATimer(stats, 'data_loading'):

                start=i*batch_size
                end=(i+1)*batch_size
                data=yourdata[start:end]
                labels=yourlabels[start:end]

            with CUDATimer(stats, 'forward'):
                outputs=yourmodel(data)

            with CUDATimer(stats, 'loss_computation'):

                loss=myloss(outputs,labels)
                myoptimizer.zero_grad()
            with CUDATimer(stats, 'backward'):
                loss.backward()

            with CUDATimer(stats, 'weight_updates'):
                myoptimizer.step()
            epoch_loss += loss.item()

        print(epoch_loss/one_run)

def evaluate(yourmodel,yourtdata,yourtlabels):
    total_batch_accuracy=0.0
    yourmodel.eval()
    with torch.no_grad():
        test_run_lenght=len(yourtdata)//batch_size
        for i in range(test_run_lenght):
            start=i*batch_size
            end=(i+1)*batch_size
            data=yourtdata[start:end]  
            labels=yourtlabels[start:end]
            total_batch=labels.size(0)
            outputs=yourmodel(data)
            _, predicted = torch.max(outputs, 1)
            correct_batch = (predicted == labels).sum().item()
            batch_accuracy = correct_batch / total_batch
            total_batch_accuracy+=batch_accuracy
    avg_t_batch_accuracy=total_batch_accuracy/test_run_lenght
    print(f"Average Batch Accuracy: {avg_t_batch_accuracy * 100:.2f}%")

if __name__ == "__main__":
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
    }
    print(f"Number of epochs: {epochs}")

    print(x_train.shape, x_test.shape, x_train.device)
    model = MLP(784,256)
    model.to("cuda")
    print(model)
    sample = x_train[:batch_size]

    output = model(sample)
    print (output.shape)

    print(model.layers)
    print(type(model.layers[0]))

    with torch.no_grad():
        for i in range(1, 4, 2)  :              
            fan_in=model.layers[i].weight.size(1)
            scale = (2.0 / fan_in) ** 0.5
            model.layers[i].weight.uniform_(-scale, scale)
            model.layers[i].bias.zero_()

    # TRAIN
    torch.cuda.synchronize()
    start = time.time()
    training(model,x_train,y_train,epochs,timing_stats)
    torch.cuda.synchronize()
    end = time.time()
    print(f"training time is ", end-start)
    # TEST
    torch.cuda.synchronize()
    start = time.time()
    evaluate(model,x_test,y_test)
    torch.cuda.synchronize()
    end = time.time()
    print(f"test time is ", end-start)
    print(timing_stats)