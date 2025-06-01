"""
LSTM Performance Test on All 300 ASL Classes
Tests LSTM architecture's learning capabilities on the complete ASL dataset.

This test focuses specifically on LSTM architecture to determine:
- Can LSTM learn from the same batch repeatedly? (overfitting test)
- Can LSTM generalize across different batches? (generalization test)
- How does LSTM perform with the full 300-class complexity?

Based on previous findings, LSTM showed the most promise for multi-batch learning,
so this test validates its performance at full scale.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TestDataset(Dataset):
    def __init__(self, data_dir, selected_classes, max_sequence_length=50, max_samples_per_class=20):
        self.data = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        
        print(f"Loading data for {len(selected_classes)} classes...")
        
        for class_name in selected_classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_name} not found")
                continue
                
            class_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
            # Limit samples per class to manage dataset size
            class_files = class_files[:max_samples_per_class]
            
            class_label = self.class_to_idx[class_name]
            
            for filename in class_files:
                try:
                    file_path = os.path.join(class_dir, filename)
                    data = np.load(file_path)
                    keypoints = data['nodes']  # Shape: (frames, 553, 3)
                    
                    if len(keypoints) == 0:
                        continue
                    
                    # Truncate or pad to max_sequence_length
                    if len(keypoints) > max_sequence_length:
                        keypoints = keypoints[:max_sequence_length]
                    else:
                        padding_needed = max_sequence_length - len(keypoints)
                        padding = np.zeros((padding_needed, keypoints.shape[1], keypoints.shape[2]))
                        keypoints = np.concatenate([keypoints, padding], axis=0)
                    
                    # Flatten to (sequence_length, features)
                    keypoints_flat = keypoints.reshape(max_sequence_length, -1)  # (50, 1659)
                    
                    self.data.append(keypoints_flat)
                    self.labels.append(class_label)
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# Removed Conv1D and Transformer models - focusing only on LSTM

class LSTMModel(nn.Module):
    """LSTM-based architecture"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
          # Use the last output
        final_output = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        final_output = self.dropout(final_output)
        output = self.classifier(final_output)
        
        return output

# Removed Conv1D and Transformer models - focusing only on LSTM

def train_model(model, train_loader, num_epochs, device, lr=0.001):
    """Train model and return loss history"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        loss_history.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return loss_history

def test_same_batch_learning(model_class, dataset, device, epochs=30, **model_kwargs):
    """Test if model can learn from the same batch repeatedly"""
    print(f"\n=== Testing Same-Batch Learning: {model_class.__name__} ===")
    
    # Get num_classes from model_kwargs
    num_classes = model_kwargs.get('num_classes', 20)
    
    # Create a small batch
    batch_size = min(32, len(dataset))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get first batch
    data_batch, labels_batch = next(iter(train_loader))
    
    # Create model
    model = model_class(**model_kwargs)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Batch size: {len(data_batch)}")
    print(f"Unique classes in batch: {len(torch.unique(labels_batch))}")
    print(f"Expected ln(num_classes) loss: {math.log(num_classes):.4f}")
    
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(data_batch.to(device))
        loss = criterion(outputs, labels_batch.to(device))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 5 == 0:
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels_batch.to(device)).float().mean().item()
                print(f'Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.3f}')
    
    final_loss = losses[-1]
    print(f"Final loss: {final_loss:.4f}")
    print(f"Can learn same batch: {'YES' if final_loss < 2.0 else 'NO'}")
    
    return losses

def test_multi_batch_learning(model_class, dataset, device, epochs=50, **model_kwargs):
    """Test if model can learn across different batches"""
    print(f"\n=== Testing Multi-Batch Learning: {model_class.__name__} ===")
    
    # Get num_classes from model_kwargs
    num_classes = model_kwargs.get('num_classes', 20)
    
    # Create data loader for multi-batch training
    batch_size = min(16, len(dataset) // 4)  # Ensure multiple batches
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Expected ln(num_classes) loss: {math.log(num_classes):.4f}")
    
    # Create model
    model = model_class(**model_kwargs)
    loss_history = train_model(model, train_loader, epochs, device)
    
    # Test final performance
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    final_loss = total_loss / len(train_loader)
    final_accuracy = correct / total
    
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final accuracy: {final_accuracy:.3f}")
    print(f"Can learn across batches: {'YES' if final_loss < 2.5 else 'NO'}")
    
    return loss_history

def plot_lstm_results(lstm_same_losses, lstm_multi_losses, num_classes):
    """Plot LSTM learning curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # LSTM Same Batch
    ax1.plot(lstm_same_losses, 'g-', linewidth=2)
    ax1.set_title('LSTM: Same Batch Learning')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.axhline(y=math.log(num_classes), color='r', linestyle='--', alpha=0.7, 
                label=f'ln({num_classes}) ≈ {math.log(num_classes):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LSTM Multi Batch
    ax2.plot(lstm_multi_losses, 'g-', linewidth=2)
    ax2.set_title('LSTM: Multi-Batch Learning')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.axhline(y=math.log(num_classes), color='r', linestyle='--', alpha=0.7, 
                label=f'ln({num_classes}) ≈ {math.log(num_classes):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'lstm_{num_classes}class_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Removed unused plotting functions - focusing only on LSTM

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all available classes from the data directory
    data_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints"
    all_classes = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    # Filter classes that have sufficient data (at least 5 samples)
    selected_classes = []
    min_samples = 5
    for class_name in all_classes:
        class_dir = os.path.join(data_dir, class_name)
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
        if len(class_files) >= min_samples:
            selected_classes.append(class_name)
    
    selected_classes = sorted(selected_classes)  # Ensure consistent ordering
    num_classes = len(selected_classes)
    print(f"Testing LSTM with {num_classes} classes (from {len(all_classes)} total available)")
    print(f"Classes with insufficient data (<{min_samples} samples): {len(all_classes) - num_classes}")
    
    # Create dataset
    dataset = TestDataset(data_dir, selected_classes, max_sequence_length=50, max_samples_per_class=20)
    
    if len(dataset) == 0:
        print("No data loaded! Check data directory.")
        return
      # Model configurations
    input_dim = 1659  # From data shape
    hidden_dim = 256  # Increased for better capacity
    
    lstm_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'num_layers': 2
    }
    
    print("\n" + "="*60)
    print(f"LSTM PERFORMANCE TEST - {num_classes} CLASSES")
    print("="*60)
    
    # Test LSTM architecture only
    print("\n🔍 TESTING LSTM ARCHITECTURE")
    print("-" * 40)
    lstm_same_losses = test_same_batch_learning(
        LSTMModel, dataset, device, epochs=30, **lstm_kwargs
    )
    lstm_multi_losses = test_multi_batch_learning(
        LSTMModel, dataset, device, epochs=50, **lstm_kwargs
    )
      # Summary
    print("\n" + "="*60)
    print(f"FINAL RESULTS SUMMARY - {num_classes} CLASSES")
    print("="*60)
    
    print(f"\nLSTM Architecture:")
    print(f"  Same Batch Final Loss: {lstm_same_losses[-1]:.4f}")
    print(f"  Multi Batch Final Loss: {lstm_multi_losses[-1]:.4f}")
    print(f"  Can learn same batch: {'✅ YES' if lstm_same_losses[-1] < 2.0 else '❌ NO'}")
    print(f"  Can learn across batches: {'✅ YES' if lstm_multi_losses[-1] < 2.5 else '❌ NO'}")
    
    # Key insight
    print(f"\n🔬 KEY INSIGHT:")
    final_loss = lstm_multi_losses[-1]
    
    if final_loss < 2.0:
        print(f"   🎯 LSTM is clearly superior - use this for your ASL recognition!")
    elif final_loss < 2.5:
        print(f"   ⚠️  LSTM shows promise but may need tuning")
    else:
        print("   ❌ LSTM struggles - data quality or task complexity issues!")
    
    print(f"\nRandom performance baseline: {math.log(num_classes):.4f}")
    
    # Plot LSTM results
    plot_lstm_results(lstm_same_losses, lstm_multi_losses, num_classes)

if __name__ == "__main__":
    main()