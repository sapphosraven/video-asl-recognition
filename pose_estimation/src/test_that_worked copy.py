"""
Advanced learning test comparing LSTM architecture on 20 ASL classes.
Tests the fundamental question: Can models learn across different batches?

Key findings from previous tests:
- LSTM: Shows promise for multi-batch learning but needs optimization

This test validates these findings at a larger scale with 20 classes and determines
which architecture is best suited for ASL recognition.
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
from tqdm import tqdm

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

def train_model(model, train_loader, val_loader, num_epochs, device, lr=0.001):
    """Train model and return train/val loss history"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (data, targets) in enumerate(epoch_bar):
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
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100 * correct / total if total > 0 else 0
            epoch_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_loss_history.append(avg_loss)
        # Validation
        model.eval()
        val_loss = 0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
        val_loss_history.append(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    return train_loss_history, val_loss_history

def test_multi_batch_learning(model_class, dataset, device, epochs=50, **model_kwargs):
    """Test if model can learn across different batches, with validation"""
    print(f"\n=== Testing Multi-Batch Learning: {model_class.__name__} ===")
    num_classes = model_kwargs.get('num_classes', 20)
    # Split dataset into train/val (80/20)
    val_split = 0.2
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    batch_size = min(16, len(train_set) // 4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"Expected ln(num_classes) loss: {math.log(num_classes):.4f}")
    model = model_class(**model_kwargs)
    train_loss_history, val_loss_history = train_model(model, train_loader, val_loader, epochs, device)
    # Final val performance
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    final_loss = total_loss / len(val_loader)
    final_accuracy = correct / total
    print(f"Final val loss: {final_loss:.4f}")
    print(f"Final val accuracy: {final_accuracy:.3f}")
    print(f"Can learn across batches: {'YES' if final_loss < 2.5 else 'NO'}")
    return train_loss_history, val_loss_history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Automatically detect all classes in the data directory
    data_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints"
    all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    all_classes = sorted([c for c in all_classes if not c.startswith('.')])
    num_classes = len(all_classes)
    print(f"Training on ALL classes ({num_classes}): {all_classes}")
    
    # Create dataset
    dataset = TestDataset(data_dir, all_classes, max_sequence_length=50, max_samples_per_class=20)
    
    if len(dataset) == 0:
        print("No data loaded! Check data directory.")
        return
    
    # Model configuration for LSTM only
    input_dim = 1659  # From data shape
    hidden_dim = 256  # Increased for better capacity
    sequence_length = 50
    lstm_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'num_layers': 2
    }
    
    print("\n" + "="*60)
    print(f"LSTM ARCHITECTURE TRAINING - {num_classes} CLASSES")
    print("="*60)
    
    # LSTM
    print("\n🔍 TRAINING LSTM ARCHITECTURE (multi-batch)")
    print("-" * 40)
    lstm_train_losses, lstm_val_losses = test_multi_batch_learning(
        LSTMModel, dataset, device, epochs=50, **lstm_kwargs
    )
    lstm_model = LSTMModel(**lstm_kwargs)
    lstm_model.load_state_dict(torch.load('lstm_temp.pth')) if os.path.exists('lstm_temp.pth') else None
    torch.save(lstm_model.state_dict(), f'lstm_{num_classes}class.pth')
    
    # Summary
    print("\n" + "="*60)
    print(f"FINAL RESULTS SUMMARY - {num_classes} CLASSES")
    print("="*60)
    print(f"LSTM Multi Batch Final Loss: {lstm_train_losses[-1]:.4f}")
    print(f"\nRandom performance baseline: {math.log(num_classes):.4f}")
    
    # Plot LSTM loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_train_losses, label='Train Loss')
    plt.plot(lstm_val_losses, label='Val Loss')
    plt.axhline(y=math.log(num_classes), color='r', linestyle='--', alpha=0.7, label=f'ln({num_classes})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'LSTM Architecture ({num_classes} Classes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'lstm_architecture_{num_classes}class.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()