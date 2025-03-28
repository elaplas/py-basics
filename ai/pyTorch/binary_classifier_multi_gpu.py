# What the script does?
# ✅ Trains a binary classifier using PyTorch on 2D points
# ✅ Exports the trained model to ONNX
# ✅ Loads the ONNX model with ONNX Runtime for inference
# ✅ Visualizes the training data & classification results

import os

# Provides torch tensor and many operations on tensors 
import torch 
# Provides neural network layers and loss functions
import torch.nn as nn
# Provides optimizers 
import torch.optim as optim
# Provides helper function to manipulate data during training and testing
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
# Helper type to enable the distribution of computation across multiple GPUs in an efficient way
from torch.nn.parallel import DistributedDataParallel as DDP
# Helper type to distribute computation across multiple devices
import torch.distributed as dist
# Helper type to launch multiple threads
import torch.multiprocessing as mp
# Provides matrix container and many helper functions to process/manipulate them
import numpy as np
# Provides helper functions to load, convert and export ONNX models
import onnx
# Inference engine to run ONNX model on different SW/HW platforms
import onnxruntime as onnxrt
# Provides visu
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## set a seed to generate deterministic set of data
torch.manual_seed(0)
## number of samples
n_samples = 1000
## generate 2D points
data = torch.randint(-5, 5, (n_samples, 2))
## generate indecies and shuffle them to pick data points randomly later
indecies = [i for i in range(n_samples)]
random.shuffle(indecies)
indecies = np.array(indecies).reshape(len(indecies), 1)
## split into training and test/validation 
n_training_samples = int(n_samples * 0.8)
n_test_samples = int(n_samples * 0.2)
x_train = data[indecies[:n_training_samples]].float()
x_test = data[indecies[n_training_samples: n_training_samples + n_test_samples]].float()
## training GT
y_train = torch.tensor([0.0 if x_train[i,:,1] < 0 else 1.0 for i in range(x_train.shape[0])])
y_train = y_train.reshape(x_train.shape[0], 1, 1)
## test GT
y_test = torch.tensor([0.0 if x_test[i,:,1] < 0 else 1.0 for i in range(x_test.shape[0])])
y_test = y_test.reshape(y_test.shape[0], 1, 1)
## Create dataloader iterables
n_epochs = 5
batch_size = 100
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
## Move to device
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
## Build dataloder for training
n_epochs = 5
batch_size = 100
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def cal_accuracy(x: torch.tensor, y: torch.tensor):
    res = 0
    for i in range(x.size()[0]):
        if abs(x[i] - y[i]) <= 0.1:
            res += 1
    res /= float(x.size()[0])
    return res

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size, train_dataset, test_dataset, n_epochs=10, batch_size=100):
    # Initialzation
    setup(rank, world_size)
    # Create distributed dataloader 
    distributed_train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, sampler=distributed_train_sampler)
    # Create model
    model = BinaryClassifier()
    model = DDP(model, device_ids=[rank])

    ## Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    ## Training process
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction_batch = model(x_batch)
            loss = loss_fn(prediction_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"..........training with {device}..............")
                print(f"epoch: {epoch} training loss: {loss}")
                print(f"epoch: {epoch} training accuracy: {cal_accuracy(prediction_batch, y_batch)}")
                prediction_test = model(test_dataset[0])
                print(f"epoch: {epoch} validation loss: {loss_fn(prediction_test, test_dataset[1])}")
                print(f"epoch: {epoch} validation accuracy: {cal_accuracy(prediction_test, test_dataset[1])}")
    # Cleanup DDP process
    dist.destroy_process_group()  


if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Get number of available GPUs
    if world_size < 2:
        print("Warning: Less than 2 GPUs available, DDP might not be beneficial.")

    # Spawn processes and pass dataset to them
    mp.spawn(train, args=(world_size, train_dataset, test_dataset), nprocs=world_size, join=True)



