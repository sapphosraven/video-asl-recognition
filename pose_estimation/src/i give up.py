"""
Advanced learning test comparing Conv1D vs LSTM vs Transformer architectures on 20 ASL classes.
Tests the fundamental question: Can models learn across different batches?

Key findings from previous tests:
- Conv1D: Can overfit same batch repeatedly but fails completely on multi-batch learning
- LSTM: Shows promise for multi-batch learning but needs optimization
- Transformer: Unknown performance - this test will reveal its capabilities

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

class Conv1DModel(nn.Module):
    """Conv1D-based architecture (from previous SimpleTGCN)"""
    def __init__(self, input_dim, hidden_dim, num_classes, sequence_length):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = x.transpose(1, 2)  # (batch, features, sequence)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

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

class TransformerModel(nn.Module):
    """Transformer-based architecture"""
    def __init__(self, input_dim, hidden_dim, num_classes, sequence_length, num_heads=8, num_layers=4):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, hidden_dim) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch, sequence, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        x = self.transformer(x)  # (batch, sequence, hidden_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification
        x = self.layer_norm(x)
        x = self.dropout(x)
        output = self.classifier(x)
        
        return output

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

def plot_three_architecture_comparison(conv1d_same, conv1d_multi, lstm_same, lstm_multi, transformer_same, transformer_multi):
    """Plot learning curves for all three architectures"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
    
    # Conv1D Same Batch
    ax1.plot(conv1d_same, 'b-', linewidth=2)
    ax1.set_title('Conv1D: Same Batch Learning')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Conv1D Multi Batch
    ax2.plot(conv1d_multi, 'b-', linewidth=2)
    ax2.set_title('Conv1D: Multi-Batch Learning')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # LSTM Same Batch
    ax3.plot(lstm_same, 'g-', linewidth=2)
    ax3.set_title('LSTM: Same Batch Learning')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # LSTM Multi Batch
    ax4.plot(lstm_multi, 'g-', linewidth=2)
    ax4.set_title('LSTM: Multi-Batch Learning')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Transformer Same Batch
    ax5.plot(transformer_same, 'r-', linewidth=2)
    ax5.set_title('Transformer: Same Batch Learning')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Transformer Multi Batch
    ax6.plot(transformer_multi, 'r-', linewidth=2)
    ax6.set_title('Transformer: Multi-Batch Learning')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss')
    ax6.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('three_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison(conv1d_same, conv1d_multi, lstm_same, lstm_multi):
    """Plot learning curves for comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Conv1D Same Batch
    ax1.plot(conv1d_same, 'b-', linewidth=2)
    ax1.set_title('Conv1D: Same Batch Learning')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Conv1D Multi Batch
    ax2.plot(conv1d_multi, 'b-', linewidth=2)
    ax2.set_title('Conv1D: Multi-Batch Learning')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # LSTM Same Batch
    ax3.plot(lstm_same, 'g-', linewidth=2)
    ax3.set_title('LSTM: Same Batch Learning')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # LSTM Multi Batch
    ax4.plot(lstm_multi, 'g-', linewidth=2)
    ax4.set_title('LSTM: Multi-Batch Learning')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.axhline(y=math.log(20), color='r', linestyle='--', alpha=0.7, label='ln(20) ≈ 3.0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('20class_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
      # 20 Classes (selecting from available words)
    selected_classes = [
        'about', 'apple', 'book', 'cat', 'dance', 'eat', 'family', 'good',
        'happy', 'house', 'like', 'man', 'new', 'play', 'school', 'table',
        'walk', 'water', 'work', 'yellow'
    ]
    
    num_classes = len(selected_classes)
    print(f"Testing with {num_classes} classes: {selected_classes}")
    
    # Create dataset
    data_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints"
    dataset = TestDataset(data_dir, selected_classes, max_sequence_length=50, max_samples_per_class=20)
    
    if len(dataset) == 0:
        print("No data loaded! Check data directory.")
        return
    
    # Model configurations
    input_dim = 1659  # From data shape
    hidden_dim = 256  # Increased for better capacity
    sequence_length = 50
    
    conv1d_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'sequence_length': sequence_length
    }
    
    lstm_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'num_layers': 2
    }
    
    transformer_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': 256,  # Keep moderate for Transformer
        'num_classes': num_classes,
        'sequence_length': sequence_length,
        'num_heads': 8,
        'num_layers': 4
    }
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ARCHITECTURE COMPARISON - 20 CLASSES")
    print("="*60)
    
    # Test Conv1D architecture
    print("\n🔍 TESTING CONV1D ARCHITECTURE")
    print("-" * 40)
    conv1d_same_losses = test_same_batch_learning(
        Conv1DModel, dataset, device, epochs=30, **conv1d_kwargs
    )
    conv1d_multi_losses = test_multi_batch_learning(
        Conv1DModel, dataset, device, epochs=50, **conv1d_kwargs
    )
    
    # Test LSTM architecture
    print("\n🔍 TESTING LSTM ARCHITECTURE")
    print("-" * 40)
    lstm_same_losses = test_same_batch_learning(
        LSTMModel, dataset, device, epochs=30, **lstm_kwargs
    )
    lstm_multi_losses = test_multi_batch_learning(
        LSTMModel, dataset, device, epochs=50, **lstm_kwargs
    )
    
    # Test Transformer architecture
    print("\n🔍 TESTING TRANSFORMER ARCHITECTURE")
    print("-" * 40)
    transformer_same_losses = test_same_batch_learning(
        TransformerModel, dataset, device, epochs=30, **transformer_kwargs
    )
    transformer_multi_losses = test_multi_batch_learning(
        TransformerModel, dataset, device, epochs=50, **transformer_kwargs
    )
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY - 20 CLASSES")
    print("="*60)
    
    print(f"\nConv1D Architecture:")
    print(f"  Same Batch Final Loss: {conv1d_same_losses[-1]:.4f}")
    print(f"  Multi Batch Final Loss: {conv1d_multi_losses[-1]:.4f}")
    print(f"  Can learn same batch: {'✅ YES' if conv1d_same_losses[-1] < 2.0 else '❌ NO'}")
    print(f"  Can learn across batches: {'✅ YES' if conv1d_multi_losses[-1] < 2.5 else '❌ NO'}")
    
    print(f"\nLSTM Architecture:")
    print(f"  Same Batch Final Loss: {lstm_same_losses[-1]:.4f}")
    print(f"  Multi Batch Final Loss: {lstm_multi_losses[-1]:.4f}")
    print(f"  Can learn same batch: {'✅ YES' if lstm_same_losses[-1] < 2.0 else '❌ NO'}")
    print(f"  Can learn across batches: {'✅ YES' if lstm_multi_losses[-1] < 2.5 else '❌ NO'}")
    
    print(f"\nTransformer Architecture:")
    print(f"  Same Batch Final Loss: {transformer_same_losses[-1]:.4f}")
    print(f"  Multi Batch Final Loss: {transformer_multi_losses[-1]:.4f}")
    print(f"  Can learn same batch: {'✅ YES' if transformer_same_losses[-1] < 2.0 else '❌ NO'}")
    print(f"  Can learn across batches: {'✅ YES' if transformer_multi_losses[-1] < 2.5 else '❌ NO'}")
    
    # Architecture ranking
    print(f"\n🏆 ARCHITECTURE RANKING (by multi-batch performance):")
    architectures = [
        ("Conv1D", conv1d_multi_losses[-1]),
        ("LSTM", lstm_multi_losses[-1]),
        ("Transformer", transformer_multi_losses[-1])
    ]
    architectures.sort(key=lambda x: x[1])  # Sort by loss (lower is better)
    
    for i, (name, loss) in enumerate(architectures, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"   {status} {i}. {name}: {loss:.4f}")
    
    # Key insight
    print(f"\n🔬 KEY INSIGHT:")
    best_arch = architectures[0][0]
    best_loss = architectures[0][1]
    
    if best_loss < 2.0:
        print(f"   🎯 {best_arch} is clearly superior - use this for your ASL recognition!")
    elif best_loss < 2.5:
        print(f"   ⚠️  {best_arch} shows promise but may need tuning")
    else:
        print("   ❌ All architectures struggle - data quality or task complexity issues!")
    
    print(f"\nRandom performance baseline: {math.log(num_classes):.4f}")
      # Plot comparison
    plot_three_architecture_comparison(
        conv1d_same_losses, conv1d_multi_losses,
        lstm_same_losses, lstm_multi_losses,
        transformer_same_losses, transformer_multi_losses
    )

if __name__ == "__main__":
    main()