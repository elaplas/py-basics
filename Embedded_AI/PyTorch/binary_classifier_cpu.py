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
# Provides helper types to handle data during training
from torch.utils.data import TensorDataset, DataLoader
# Provides matrix container and many helper functions to process/manipulate them
import numpy as np
# Provides helper functions to load, convert and export ONNX models
import onnx
# Inference engine to run ONNX model on different SW/HW platforms
import onnxruntime as onnxrt
# Provides visu
import matplotlib.pyplot as plt

torch.manual_seed(0)
n_samples = 1000
data = torch.randint(-5, 5, (n_samples, 2))
n_training_samples = int(n_samples * 0.8)
n_test_samples = int(n_samples * 0.2)
x_train = data[:n_training_samples].float()
x_test = data[n_training_samples: n_training_samples + n_test_samples].float()
## training GT
condition = x_train[:, 0] < 0
condition = condition.reshape((condition.size()[0], 1))
y_train = condition.float()
## test GT
condition = x_test[:, 0] < 0
condition = condition.reshape((condition.size()[0], 1))
y_test = condition.float()
## Create dataloader iterables
n_epochs = 5
batch_size = 100
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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

model = BinaryClassifier()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        prediction_batch = model(x_batch)
        loss = loss_fn(prediction_batch, y_batch )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("........................")
        print(f"training loss: {loss}")
        print(f"training accuracy: {cal_accuracy(prediction_batch, y_batch)}")
        prediction_test = model(x_test)
        print(f"validation loss: {loss_fn(prediction_test, y_test)}")
        print(f"validation accuracy: {cal_accuracy(prediction_test, y_test)}")

# Export to onnx format 
sample_input = torch.randint(-5, 5, (1, 2)).float() # Type and shape of sample input is decisive
model_path = "./binary_classifier.onnx"
torch.onnx.export(model, sample_input, model_path, input_names=["input"], output_names=["output"])

# Instantiate and config inference session
ort_session = onnxrt.InferenceSession(model_path)

# Inferencing samples one-by-one
inf_res = []
for i in range(x_test.size()[0]):
    x_i = x_test.numpy()[i].reshape(1,2)
    outputs_i = ort_session.run(None, {"input": x_i})
    inf_res.append(outputs_i[0].tolist())

inf_acc = cal_accuracy(torch.tensor(inf_res), y_test)
print(".................")
print(f"inference accuracy: {cal_accuracy(torch.tensor(inf_res), y_test)}")


