#!/usr/bin/env python
# coding: utf-8

# # Temporal GCN Workflow for ASL Pose Sequences
# 
# This notebook implements the full TGCN pipeline in self-contained phases:
# 
# 1. Core TGCN architecture
# 2. Dataset loader & preprocessing
# 3. Training loop with checkpoints
# 4. Evaluation metrics & inference
# 5. Hyperparameter tuning & optimization
# 
# All intermediate models, logs, and results are saved to disk to prevent data loss if the notebook or IDE crashes.
# 

# ## Phase 1: Core TGCN Architecture
# 
# Define graph convolution and temporal graph convolution layers using PyTorch and PyTorch Geometric.
# 

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from mediapipe.python.solutions import pose_connections, hands_connections
import os
import pickle
from tqdm import tqdm

# Build static adjacency matrix (75x75) for MediaPipe pose + hands
NUM_POSE, NUM_HAND = 33, 21
TOTAL_NODES = NUM_POSE + 2 * NUM_HAND

# Get connections from MediaPipe
POSE_CONNECTIONS = pose_connections.POSE_CONNECTIONS
HAND_CONNECTIONS = hands_connections.HAND_CONNECTIONS

# Build edge list
edges = set()
# Pose connections (0-32)
for u, v in POSE_CONNECTIONS:
    edges.add((u, v))
    edges.add((v, u))

# Left hand connections (33-53)
off1 = NUM_POSE
for u, v in HAND_CONNECTIONS:
    edges.add((off1 + u, off1 + v))
    edges.add((off1 + v, off1 + u))

# Right hand connections (54-74)
off2 = NUM_POSE + NUM_HAND
for u, v in HAND_CONNECTIONS:
    edges.add((off2 + u, off2 + v))
    edges.add((off2 + v, off2 + u))

# Convert to edge_index format for PyTorch Geometric
edge_list = list(edges)
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

print(f"Graph structure: {TOTAL_NODES} nodes, {len(edge_list)} edges")
print(f"Edge index shape: {edge_index.shape}")

class TGCNLayer(nn.Module):
    """Temporal Graph Convolutional Layer"""
    def __init__(self, in_channels, out_channels, K=3):
        super(TGCNLayer, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Graph convolution
        self.gcn = GCNConv(in_channels, out_channels)
        
        # Temporal convolution (1D conv over time dimension)
        self.temporal_conv = nn.Conv1d(out_channels, out_channels, kernel_size=K, padding=K//2)
        
        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: [batch_size * seq_len, num_nodes, in_channels]
            edge_index: [2, num_edges]
        Returns:
            x: [batch_size * seq_len, num_nodes, out_channels]
        """
        batch_seq, num_nodes, _ = x.shape
        
        # Apply graph convolution
        x_flat = x.reshape(-1, self.in_channels)  # [batch_seq * num_nodes, in_channels]
        
        # Create expanded edge_index for batched graphs
        batch_size = batch_seq
        edge_indices = []
        for i in range(batch_size):
            edge_idx = edge_index + i * num_nodes
            edge_indices.append(edge_idx)
        
        edge_index_batch = torch.cat(edge_indices, dim=1)
        
        # Apply GCN
        x_gcn = self.gcn(x_flat, edge_index_batch)  # [batch_seq * num_nodes, out_channels]
        x_gcn = x_gcn.reshape(batch_seq, num_nodes, self.out_channels)
        
        # Apply temporal convolution over sequence dimension
        # Reshape for temporal conv: [batch, channels, sequence]
        x_temp = x_gcn.permute(1, 2, 0)  # [num_nodes, out_channels, batch_seq]
        
        # Apply temporal conv to each node independently
        x_out = []
        for node_idx in range(num_nodes):
            node_features = x_temp[node_idx]  # [out_channels, batch_seq]
            node_conv = self.temporal_conv(node_features.unsqueeze(0)).squeeze(0)
            x_out.append(node_conv)
        
        x_out = torch.stack(x_out, dim=0)  # [num_nodes, out_channels, batch_seq]
        x_out = x_out.permute(2, 0, 1)  # [batch_seq, num_nodes, out_channels]
        
        # Apply normalization and activation
        x_out = x_out.permute(0, 2, 1)  # [batch_seq, out_channels, num_nodes]
        x_out = self.batch_norm(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout(x_out)
        x_out = x_out.permute(0, 2, 1)  # [batch_seq, num_nodes, out_channels]
        
        return x_out

class TGCN(nn.Module):
    """Temporal Graph Convolutional Network for ASL Recognition"""
    def __init__(self, num_nodes, in_features, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # TGCN layers
        self.tgcn_layers = nn.ModuleList()
        
        # First layer
        self.tgcn_layers.append(TGCNLayer(in_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.tgcn_layers.append(TGCNLayer(hidden_dim, hidden_dim))
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: [batch_size, seq_len, num_nodes, in_features]
            edge_index: [2, num_edges]
        Returns:
            output: [batch_size, num_classes]
        """
        batch_size, seq_len, num_nodes, in_features = x.shape
        
        # Reshape for processing: [batch_size * seq_len, num_nodes, in_features]
        x = x.reshape(batch_size * seq_len, num_nodes, in_features)
        
        # Apply TGCN layers
        for layer in self.tgcn_layers:
            x = layer(x, edge_index)
        
        # Reshape back: [batch_size, seq_len, num_nodes, hidden_dim]
        x = x.reshape(batch_size, seq_len, num_nodes, -1)
        
        # Global pooling over nodes: [batch_size, seq_len, hidden_dim]
        x = x.mean(dim=2)
        
        # Temporal pooling: [batch_size, hidden_dim, seq_len] -> [batch_size, hidden_dim, 1]
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Test the model structure
print("\n=== TGCN Architecture Test ===")
model = TGCN(num_nodes=TOTAL_NODES, in_features=3, hidden_dim=64, num_classes=10)
test_input = torch.randn(2, 50, TOTAL_NODES, 3)  # batch=2, seq=50, nodes=75, features=3
test_output = model(test_input, edge_index)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print("✓ TGCN architecture implemented successfully!")


# ## Phase 2: Dataset Loader & Preprocessing
# 
# Load NPZ keypoint sequences, pad/truncate to fixed length, and build PyTorch Geometric Data objects.
# 

# In[4]:


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

class PoseSequenceDataset(Dataset):
    """Dataset for pose keypoint sequences from NPZ files"""
    
    def __init__(self, data_dir, max_seq_len=80, split='train', test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.split = split
        
        # Find all NPZ files
        self.files = []
        self.labels = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        print(f"Loading dataset from: {data_dir}")
        
        # Scan directory structure
        word_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found {len(word_dirs)} word categories")
        
        # Build label mapping
        for idx, word in enumerate(sorted(word_dirs)):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.num_classes = len(self.word_to_idx)
        print(f"Number of classes: {self.num_classes}")
        
        # Collect all files with labels
        file_label_pairs = []
        for word, label_idx in self.word_to_idx.items():
            word_dir = os.path.join(data_dir, word)
            npz_files = glob.glob(os.path.join(word_dir, "*.npz"))
            
            for file_path in npz_files:
                file_label_pairs.append((file_path, label_idx))
        
        print(f"Total files found: {len(file_label_pairs)}")
        
        if len(file_label_pairs) == 0:
            raise ValueError(f"No NPZ files found in {data_dir}")
        
        # Split into train/test
        files, labels = zip(*file_label_pairs)
        
        if len(files) > 1:
            train_files, test_files, train_labels, test_labels = train_test_split(
                files, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
        else:
            # Edge case: only one file
            train_files, test_files = files, files
            train_labels, test_labels = labels, labels
        
        if split == 'train':
            self.files = list(train_files)
            self.labels = list(train_labels)
        else:
            self.files = list(test_files)
            self.labels = list(test_labels)
        
        print(f"{split.upper()} split: {len(self.files)} files")
        
        # Class distribution
        class_counts = defaultdict(int)
        for label in self.labels:
            class_counts[self.idx_to_word[label]] += 1
        
        print(f"Class distribution in {split}:")
        for word, count in sorted(class_counts.items()):
            print(f"  {word}: {count} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Load NPZ file
            data = np.load(file_path)
            
            # Handle different possible keys
            if 'keypoints' in data:
                keypoints = data['keypoints']  # Shape: (seq_len, num_nodes, 3)
            elif 'pose_keypoints' in data:
                keypoints = data['pose_keypoints']
            elif 'nodes' in data:
                keypoints = data['nodes']
            else:
                # Try to find any array that looks like keypoints
                arrays = [key for key in data.keys() if len(data[key].shape) == 3]
                if arrays:
                    keypoints = data[arrays[0]]
                else:
                    raise ValueError(f"No suitable keypoint data found in {file_path}")
            
            # Ensure proper shape
            if len(keypoints.shape) == 3:
                seq_len, num_nodes, num_features = keypoints.shape
            else:
                raise ValueError(f"Invalid keypoint shape: {keypoints.shape}")
            
            # Handle sequence length
            if seq_len > self.max_seq_len:
                # Downsample sequence
                indices = np.linspace(0, seq_len - 1, self.max_seq_len, dtype=int)
                keypoints = keypoints[indices]
            elif seq_len < self.max_seq_len:
                # Pad sequence
                padding = np.zeros((self.max_seq_len - seq_len, num_nodes, num_features))
                keypoints = np.concatenate([keypoints, padding], axis=0)
            
            # Normalize keypoints to [-1, 1] range
            keypoints = keypoints.astype(np.float32)
            
            # Remove NaN values
            keypoints = np.nan_to_num(keypoints, nan=0.0)
            
            # Basic normalization (optional - can be improved)
            keypoints = np.clip(keypoints, -2.0, 2.0)
            
            return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor as fallback
            zero_keypoints = torch.zeros((self.max_seq_len, TOTAL_NODES, 3), dtype=torch.float32)
            return zero_keypoints, torch.tensor(label, dtype=torch.long)

# Load datasets
data_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints"

print("\n=== Loading Datasets ===")
train_dataset = PoseSequenceDataset(data_dir, max_seq_len=80, split='train', test_size=0.2)
val_dataset = PoseSequenceDataset(data_dir, max_seq_len=80, split='val', test_size=0.2)

# Create data loaders
batch_size = 8  # Start with smaller batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\nDataLoaders created:")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Batch size: {batch_size}")

# Test data loading
print("\n=== Testing Data Loading ===")
try:
    sample_batch = next(iter(train_loader))
    sample_x, sample_y = sample_batch
    print(f"Sample batch shape: {sample_x.shape}")  # Should be [batch_size, seq_len, nodes, features]
    print(f"Sample labels shape: {sample_y.shape}")  # Should be [batch_size]
    print(f"Label range: {sample_y.min().item()} - {sample_y.max().item()}")
    print(f"Keypoint value range: {sample_x.min().item():.3f} - {sample_x.max().item():.3f}")
    print("✓ Data loading successful!")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

# Save label mappings
label_mapping = {
    'word_to_idx': train_dataset.word_to_idx,
    'idx_to_word': train_dataset.idx_to_word,
    'num_classes': train_dataset.num_classes
}

with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)

print(f"\nLabel mapping saved. Number of classes: {train_dataset.num_classes}")


# ## Phase 3: Training Loop with Checkpoints
# 
# Implement training & validation loops, saving model checkpoints at each epoch.
# 

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Training configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model parameters
hidden_dim = 128
num_classes = train_dataset.num_classes
learning_rate = 0.001
num_epochs = 50
patience = 10  # Early stopping patience

# Initialize model
model = TGCN(
    num_nodes=TOTAL_NODES,
    in_features=3,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=3,
    dropout=0.3
).to(device)

print(f"\nModel initialized:")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Move edge_index to device
edge_index_device = edge_index.to(device)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

# Early stopping variables
best_val_loss = float('inf')
best_val_acc = 0.0
patience_counter = 0
best_model_path = 'best_tgcn_model.pt'

def calculate_accuracy(outputs, labels):
    """Calculate accuracy from model outputs and true labels"""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def train_epoch(model, train_loader, criterion, optimizer, device, edge_index):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data, edge_index)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        acc = calculate_accuracy(outputs, labels)
        total_correct += acc * labels.size(0)
        total_samples += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def validate_epoch(model, val_loader, criterion, device, edge_index):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validation", leave=False):
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data, edge_index)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            total_correct += acc * labels.size(0)
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# Training loop
print(f"\n=== Starting Training ===")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Device: {device}")
print("-" * 50)

start_time = time.time()

for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress", unit="epoch"):
    epoch_start = time.time()
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, edge_index_device)
    
    # Validation
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, edge_index_device)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    # Print epoch results
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch} Results:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Time: {epoch_time:.2f}s")
    
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        checkpoint_path = f'tgcn_checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Early stopping and best model saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0
        
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_config': {
                'num_nodes': TOTAL_NODES,
                'in_features': 3,
                'hidden_dim': hidden_dim,
                'num_classes': num_classes,
                'num_layers': 3,
                'dropout': 0.3
            }
        }, best_model_path)
        print(f"  ✓ New best model saved! Val Acc: {val_acc:.4f}")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{patience}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch} epochs")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        break
    
    print("-" * 50)

total_time = time.time() - start_time
print(f"\n=== Training Completed ===")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best model saved as: {best_model_path}")

# Plot training history
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Learning rate plot
plt.subplot(1, 3, 3)
plt.plot(history['lr'], label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Training completed successfully!")

# Execute this cell to start training
print('Starting training execution...')


# ## Phase 4: Evaluation & Inference
# 
# Compute accuracy, F1-score, and confusion matrix on validation/test split.
# 

# In[ ]:


import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict

# Load the best model
print("=== Model Evaluation ===")
print("Loading best model...")

# Load model checkpoint
checkpoint = torch.load(best_model_path, map_location=device)
model_config = checkpoint['model_config']

# Recreate model with saved configuration
eval_model = TGCN(
    num_nodes=model_config['num_nodes'],
    in_features=model_config['in_features'],
    hidden_dim=model_config['hidden_dim'],
    num_classes=model_config['num_classes'],
    num_layers=model_config['num_layers'],
    dropout=model_config['dropout']
).to(device)

# Load trained weights
eval_model.load_state_dict(checkpoint['model_state_dict'])
eval_model.eval()

print(f"Model loaded from epoch {checkpoint['epoch']}")
print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")

def evaluate_model(model, data_loader, device, edge_index, dataset_name="Test"):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nEvaluating on {dataset_name} set...")
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data, edge_index)
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Processed {batch_idx}/{len(data_loader)} batches")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(data_loader)
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'loss': avg_loss
    }

# Evaluate on validation set
val_results = evaluate_model(eval_model, val_loader, device, edge_index_device, "Validation")

# Evaluate on training set (subset for speed)
train_subset_size = min(len(train_loader), 50)  # Limit to 50 batches for speed
train_subset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    sampler=torch.utils.data.SubsetRandomSampler(range(0, train_subset_size * batch_size))
)
train_results = evaluate_model(eval_model, train_subset_loader, device, edge_index_device, "Training (subset)")

# Detailed classification report
print("\n=== Detailed Classification Report ===")
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

idx_to_word = {int(k): v for k, v in label_mapping['idx_to_word'].items()}
target_names = [idx_to_word[i] for i in range(len(idx_to_word))]

print("\nValidation Set Classification Report:")


# ## Phase 5: Hyperparameter Tuning (Optuna Stub)
# 
# Outline hyperparameter search and early stopping.
# 

# In[ ]:


import optuna

def objective(trial):
    # stub: implement search over hidden size, lr, num layers
    return 0.0

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(study.best_params)

