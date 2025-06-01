"""
Standalone LSTM Training Script for 300-Class ASL Recognition
Based on the working test_that_worked.py template but scaled for full training.

This script trains an LSTM model on all available ASL classes with:
- Comprehensive data loading and preprocessing
- Train/validation split with proper evaluation
- Mixed precision training for efficiency
- Model checkpointing and saving
- Progress tracking and visualization
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
from tqdm import tqdm
import json

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ASLDataset(Dataset):
    """Dataset class for loading ASL keypoint data"""
    def __init__(self, data_dir, selected_classes=None, max_sequence_length=50, 
                 max_samples_per_class=None, min_samples_per_class=5):
        self.data = []
        self.labels = []
        self.class_names = []
        
        # Discover all available classes if not specified
        if selected_classes is None:
            all_classes = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
            # Filter classes with sufficient samples
            selected_classes = []
            for class_name in all_classes:
                class_dir = os.path.join(data_dir, class_name)
                class_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
                if len(class_files) >= min_samples_per_class:
                    selected_classes.append(class_name)
        
        selected_classes = sorted(selected_classes)  # Ensure consistent ordering
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        self.class_names = selected_classes
        
        print(f"Loading data for {len(selected_classes)} classes...")
        print(f"Max samples per class: {max_samples_per_class or 'unlimited'}")
        
        class_sample_counts = {}
        
        for class_name in selected_classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_name} not found")
                continue
                
            class_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
            
            # Limit samples per class if specified
            if max_samples_per_class:
                class_files = class_files[:max_samples_per_class]
            
            class_label = self.class_to_idx[class_name]
            samples_loaded = 0
            
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
                    samples_loaded += 1
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
            
            class_sample_counts[class_name] = samples_loaded
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"\nDataset Summary:")
        print(f"Total samples loaded: {len(self.data)}")
        print(f"Number of classes: {len(selected_classes)}")
        print(f"Data shape: {self.data.shape}")
        print(f"Samples per class: min={min(class_sample_counts.values())}, "
              f"max={max(class_sample_counts.values())}, "
              f"avg={np.mean(list(class_sample_counts.values())):.1f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
    
    def get_class_names(self):
        return self.class_names

class LSTMModel(nn.Module):
    """LSTM-based architecture for ASL recognition"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        final_output = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        # Apply normalization and dropout
        final_output = self.layer_norm(final_output)
        final_output = self.dropout(final_output)
        output = self.classifier(final_output)
        
        return output

def create_data_loaders(dataset, batch_size=32, val_split=0.2, test_split=0.1):
    """Create train, validation, and test data loaders"""
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        range(len(dataset)), 
        test_size=test_split, 
        random_state=42, 
        stratify=dataset.labels
    )
    
    # Second split: separate train and validation from remaining data
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=val_split/(1-test_split),  # Adjust for already removed test set
        random_state=42, 
        stratify=dataset.labels[train_val_indices]
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Data splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples") 
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100 * correct / total
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (len(progress_bar) - len(val_loader) + progress_bar.n + 1)
            accuracy = 100 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    return total_loss / len(val_loader), correct / total

def train_model(model, train_loader, val_loader, num_epochs, device, 
                lr=0.001, save_dir="checkpoints"):
    """Complete training loop with checkpointing"""
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
      # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Checkpoints will be saved to: {save_dir}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
          # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate changes
        if epoch > 0 and history['lr'][-1] != current_lr:
            print(f"Learning rate reduced to: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, checkpoint_path)
            print(f"📁 Checkpoint saved: {checkpoint_path}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history, best_model_path

def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)
            
            with autocast():
                outputs = model(data)
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    if len(class_names) <= 50:  # Only show detailed report for reasonable number of classes
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(f"Classification report omitted (too many classes: {len(class_names)})")
    
    return accuracy, y_true, y_pred

def plot_training_history(history, save_path="training_history.png"):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Acc')
    ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning Rate
    ax3.plot(epochs, history['lr'], 'g-')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Loss zoom (last 50% of training)
    mid_point = len(epochs) // 2
    ax4.plot(epochs[mid_point:], history['train_loss'][mid_point:], 'b-', label='Train Loss')
    ax4.plot(epochs[mid_point:], history['val_loss'][mid_point:], 'r-', label='Val Loss')
    ax4.set_title('Loss (Second Half of Training)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to: {save_path}")

def save_training_config(config, save_path="training_config.json"):
    """Save training configuration"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to: {save_path}")

def main():
    # Configuration
    config = {        'data_dir': r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints",
        'max_sequence_length': 50,
        'max_samples_per_class': None,  # Use all available samples
        'min_samples_per_class': 5,    # Reduced from 10 to include more classes
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'hidden_dim': 512,
        'num_layers': 3,
        'dropout': 0.3,
        'val_split': 0.15,
        'test_split': 0.15,
        'save_dir': "lstm_300_class_training"
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Save configuration
    os.makedirs(config['save_dir'], exist_ok=True)
    save_training_config(config, os.path.join(config['save_dir'], 'training_config.json'))
    
    # Load dataset
    print("Loading ASL dataset...")
    dataset = ASLDataset(
        config['data_dir'],
        max_sequence_length=config['max_sequence_length'],
        max_samples_per_class=config['max_samples_per_class'],
        min_samples_per_class=config['min_samples_per_class']
    )
    
    if len(dataset) == 0:
        print("❌ No data loaded! Check data directory.")
        return
    
    class_names = dataset.get_class_names()
    num_classes = len(class_names)
    input_dim = dataset.data.shape[2]  # Features per timestep
    
    print(f"\nTraining configuration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Sequence length: {config['max_sequence_length']}")
    print(f"  Total samples: {len(dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, 
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        test_split=config['test_split']
    )
    
    # Create model
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    model.to(device)
    
    # Train model
    history, best_model_path = train_model(
        model, train_loader, val_loader, 
        num_epochs=config['num_epochs'],
        device=device,
        lr=config['learning_rate'],
        save_dir=config['save_dir']
    )
    
    # Load best model for evaluation
    print(f"\nLoading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_accuracy, y_true, y_pred = evaluate_model(model, test_loader, device, class_names)
    
    # Plot training history
    plot_path = os.path.join(config['save_dir'], 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Save final results
    results = {
        'test_accuracy': test_accuracy,
        'best_val_accuracy': checkpoint['val_acc'],
        'num_classes': num_classes,
        'total_samples': len(dataset),
        'model_parameters': total_params
    }
    
    results_path = os.path.join(config['save_dir'], 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Best Val Accuracy: {checkpoint['val_acc']:.4f} ({checkpoint['val_acc']*100:.2f}%)")
    print(f"   Classes trained: {num_classes}")
    print(f"📁 All results saved to: {config['save_dir']}")

if __name__ == "__main__":
    main()
