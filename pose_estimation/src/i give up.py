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
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json

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

# LSTM Model
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

def plot_lstm_results(lstm_same_losses, lstm_multi_losses, num_classes, model_name):
    """Plot LSTM learning curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Same Batch
    ax1.plot(lstm_same_losses, 'g-', linewidth=2)
    ax1.set_title(f'{model_name}: Same Batch Learning')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.axhline(y=math.log(num_classes), color='r', linestyle='--', alpha=0.7, 
                label=f'ln({num_classes}) ≈ {math.log(num_classes):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Multi Batch
    ax2.plot(lstm_multi_losses, 'g-', linewidth=2)
    ax2.set_title(f'{model_name}: Multi-Batch Learning')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.axhline(y=math.log(num_classes), color='r', linestyle='--', alpha=0.7, 
                label=f'ln({num_classes}) ≈ {math.log(num_classes):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_{num_classes}class_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model_with_early_stopping(model, train_loader, val_loader, device, lr=0.001, max_epochs=100, patience=10, save_path=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
        for data, targets in train_bar:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=train_correct/train_total)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False)
            for data, targets in val_bar:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                val_bar.set_postfix(loss=loss.item(), acc=val_correct/val_total)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  [Checkpoint] Model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}, val acc {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f} (epoch {best_epoch+1})")
                break
    
    return history, best_val_loss, best_epoch

def evaluate_model(model, data_loader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        eval_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for data, targets in eval_bar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate running accuracy
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            eval_bar.set_postfix(acc=correct/total)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate final accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Generate classification report and confusion matrix
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return report, cm

def random_hyperparam_search(param_grid, n_trials=10):
    """Randomly sample hyperparameters from grid."""
    for _ in range(n_trials):
        yield {k: random.choice(v) for k, v in param_grid.items()}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints"
    all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    min_samples = 5
    selected_classes = []
    for class_name in all_classes:
        class_dir = os.path.join(data_dir, class_name)
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
        if len(class_files) >= min_samples:
            selected_classes.append(class_name)
    selected_classes = sorted(selected_classes)
    num_classes = len(selected_classes)
    print(f"Found {num_classes} classes with >= {min_samples} samples.")
    dataset = TestDataset(data_dir, selected_classes, max_sequence_length=50, max_samples_per_class=20)
    if len(dataset) == 0:
        print("No data loaded! Check data directory.")
        return
    # Split dataset
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42, stratify=dataset.labels[train_idx])
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=32, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=32, shuffle=False)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")    # Hyperparameter search - Updated with better ranges
    param_grid = {
        'hidden_dim': [128, 256, 384, 512],  # Reduced range, focus on effective sizes
        'num_layers': [2, 3],  # Deeper networks for complex 300-class problem
        'dropout': [0.3, 0.4, 0.5],  # Higher dropout for regularization
        'lr': [0.001, 0.0005, 0.002, 0.003]  # Much higher learning rates
    }
    best_val_loss = float('inf')
    best_config = None
    best_model_path = os.path.join(os.path.dirname(__file__), '../models/best_lstm.pth')
    best_config_path = os.path.join(os.path.dirname(__file__), '../models/best_lstm_config.json')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)    # --- RETRAIN WITH BETTER HYPERPARAMETERS ---
    force_tune = True  # Force retraining with better hyperparameters
    if (not force_tune and os.path.exists(best_model_path) and os.path.exists(best_config_path)):
        print("Best model and config found. Skipping hyperparameter tuning.")
        # Load existing config
        with open(best_config_path, 'r') as f:
            best_config = json.load(f)
    else:
        print("Starting hyperparameter search with improved ranges...")
        for i, params in enumerate(random_hyperparam_search(param_grid, n_trials=20)):
            print(f"\nTrial {i+1}/20: {params}")
            model = LSTMModel(input_dim=1659, hidden_dim=params['hidden_dim'], num_classes=num_classes, num_layers=params['num_layers'])
            model.dropout.p = params['dropout']
            
            # Extended training with more epochs for complex 300-class problem
            history, val_loss, best_epoch = train_model_with_early_stopping(
                model, train_loader, val_loader, device,
                lr=params['lr'], max_epochs=50, patience=15, save_path=best_model_path
            )
            
            print(f"Trial {i+1} finished. Best val loss: {val_loss:.4f} at epoch {best_epoch+1}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = params.copy()
                print(f"[Best so far] New best config: {best_config}")
                # Save best config to JSON
                with open(best_config_path, 'w') as f:
                    json.dump(best_config, f, indent=2)
    print(f"\nBest hyperparameters: {best_config if best_config else '[Loaded from file]'}")
    print(f"Loading best model from {best_model_path}")
    # Load best config from JSON
    with open(best_config_path, 'r') as f:
        best_config = json.load(f)
    best_model = LSTMModel(input_dim=1659, hidden_dim=best_config['hidden_dim'], num_classes=num_classes, num_layers=best_config['num_layers'])
    best_model.dropout.p = best_config['dropout']
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)    # Evaluation
    print("\nEvaluating best model on test set...")
    report, cm = evaluate_model(best_model, test_loader, device, selected_classes)
    
    # Save results to files (without printing large report to console)
    np.save(os.path.join(os.path.dirname(best_model_path), 'confusion_matrix.npy'), cm)
    with open(os.path.join(os.path.dirname(best_model_path), 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print("Classification report and confusion matrix saved to files.")
    print(f"Overall accuracy: {report.get('accuracy', 'N/A'):.4f}")
    print(f"Macro avg F1-score: {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
    print(f"Weighted avg F1-score: {report.get('weighted avg', {}).get('f1-score', 'N/A'):.4f}")

if __name__ == "__main__":
    main()