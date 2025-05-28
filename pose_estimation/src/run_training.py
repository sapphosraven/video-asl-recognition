import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the previously created variables from the notebook execution
import sys
sys.path.append('.')

# Training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Re-import the model and data if needed
print("Setting up training...")

# This will be executed after loading all the necessary components from the notebook
print("Ready to start training TGCN model!")
