# What the script does?
# ✅ Trains a binary classifier using PyTorch on 2D points
# ✅ Exports the trained model to ONNX
# ✅ Loads the ONNX model with ONNX Runtime for inference
# ✅ Visualizes the training data & classification results

# Provides torch tensor and many helper functions e.g. to operate on tensors 
import torch 
# Provides neural network layers and loss functions
import torch.nn
# Provides optimizers 
import torch.optim as optim
# Provides matrix container and many helper functions to process/manipulate them
import numpy as np
# Provides helper functions to load, convert and export ONNX models
import onnx
# Inference engine to run ONNX model on different SW/HW platforms
import onnxruntime
# Provides visu
import matplotlib.pyplot as plt

