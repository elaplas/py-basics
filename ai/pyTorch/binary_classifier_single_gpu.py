# What the script does?
# ✅ Trains a binary classifier using PyTorch on 2D points
# ✅ Exports the trained model to ONNX
# ✅ Loads the ONNX model with ONNX Runtime for inference
# ✅ Visualizes the training data & classification results

# Provides torch tensor and many operations on tensors 
import torch 
# Provides neural network layers and loss functions
import torch.nn as nn
# Provides optimizers 
import torch.optim as optim
# Provides helper function to manipulate data during training and testing
from torch.utils.data import TensorDataset, DataLoader
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
torch.manual_seed(0)
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
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

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


## Create model and Move it to device
model = BinaryClassifier()
model = model.to(device)

## Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

## Training process
for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        prediction_batch = model(x_batch)
        loss = loss_fn(prediction_batch, y_batch )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"..........training with {device}..............")
        print(f"training loss: {loss}")
        print(f"training accuracy: {cal_accuracy(prediction_batch, y_batch)}")
        prediction_test = model(x_test)
        print(f"validation loss: {loss_fn(prediction_test, y_test)}")
        print(f"validation accuracy: {cal_accuracy(prediction_test, y_test)}")

# Export to onnx format 
sample_input = torch.randint(-5, 5, (1, 2)).float().to(device) # Type and shape of sample input is decisive
model_path = "./binary_classifier.onnx"
torch.onnx.export(model, sample_input, model_path, input_names=["input"], output_names=["output"])

# Set provider
providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

# Instantiate and config inference session
ort_session = onnxrt.InferenceSession(model_path, providers = providers)

# Inferencing samples one-by-one
inf_res = []
for i in range(x_test.size()[0]):
    x_i = x_test.cpu().numpy()[i].reshape(1,2)
    outputs_i = ort_session.run(None, {"input": x_i})
    inf_res.append(outputs_i[0].tolist())

inf_acc = cal_accuracy(torch.tensor(inf_res, device=device), y_test)
print(".................")
print(f"inference accuracy: {inf_acc}")


# CUDA Optimization Explained
# Parallelized Computation and Levels of Parallelism
# 1️⃣ GPU Parallelism in Tensor Computations (Low-Level)
# 
# PyTorch automatically parallelizes tensor operations using CUDA on GPUs.
# Operations like matrix multiplications, activations (ReLU, sigmoid), and loss computations are efficiently executed in parallel.
# Each layer in the neural network (fully connected layers) is essentially a matrix multiplication, which maps perfectly to GPU architectures.
# 2️⃣ Mini-batch Processing (Mid-Level)
# 
# Instead of training on one sample at a time, we use mini-batches (size=100), which speeds up training.
# Each mini-batch is processed in parallel across GPU cores, reducing data transfer overhead.
# 3️⃣ CUDA Kernel Execution (Hardware-Level)
# 
# Each PyTorch function (like .to(device), .backward(), and .step()) calls optimized CUDA kernels behind the scenes.
# These kernels distribute computations over thousands of CUDA cores, accelerating gradient computation and updates.
# 4️⃣ Asynchronous Execution & Tensor Operations
# 
# PyTorch executes operations asynchronously on GPUs, meaning computations are queued and executed in parallel without waiting for previous operations to finish.
# No manual parallelization is required—PyTorch automatically optimizes memory transfers and execution order.

# Inferencing on GPU
# By default, ONNX Runtime always places input(s) and output(s) on CPU unless explicitly bound to GPU memory
# Inputs/outputs can be moved on GPU with IO binding

# An example of binding
# Prepare input/output OrtValues on GPU

##### x_ort = ort.OrtValue.ortvalue_from_numpy(x_i, 'cuda', 0)
##### output_ort = ort.OrtValue.ortvalue_from_shape([1, 1], np.float32, 'cuda', 0)
##### 
##### # Bind and run
##### io_binding = ort_session.io_binding()
##### io_binding.bind_input('input', x_ort.device_name(), 0, np.float32, x_ort.shape(), x_ort.data_ptr())
##### io_binding.bind_output('output', output_ort.device_name(), 0, np.float32, output_ort.shape(), output_ort.data_ptr())
##### ort_session.run_with_iobinding(io_binding)
##### 
##### # Access output on GPU without CPU copy
##### gpu_output = io_binding.get_outputs()[0]



