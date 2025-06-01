"""
Quick LSTM Test - Test on 20 classes first to verify training works
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
            class_files = class_files[:max_samples_per_class]
            
            for file in class_files:
                file_path = os.path.join(class_dir, file)
                try:
                    data = np.load(file_path)
                    keypoints = data['keypoints']
                    
                    if len(keypoints.shape) == 3:
                        frames, num_points, coords = keypoints.shape
                        flattened = keypoints.reshape(frames, -1)
                        
                        if flattened.shape[1] == 1659:
                            if frames > max_sequence_length:
                                flattened = flattened[:max_sequence_length]
                            elif frames < max_sequence_length:
                                padding = np.zeros((max_sequence_length - frames, 1659))
                                flattened = np.vstack([flattened, padding])
                            
                            self.data.append(flattened)
                            self.labels.append(self.class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

def train_quick_test(model, train_loader, val_loader, device, lr=0.001, max_epochs=20):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  [Best] New best val accuracy: {best_val_acc:.4f}")
    
    return best_val_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints"
    
    # Get first 20 classes with sufficient data
    all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    selected_classes = []
    
    for class_name in sorted(all_classes):
        class_dir = os.path.join(data_dir, class_name)
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.npz')]
        if len(class_files) >= 5:
            selected_classes.append(class_name)
            if len(selected_classes) >= 20:  # Limit to 20 classes
                break
    
    num_classes = len(selected_classes)
    print(f"Testing with {num_classes} classes: {selected_classes[:5]}...")
    
    # Load dataset
    dataset = TestDataset(data_dir, selected_classes, max_sequence_length=50, max_samples_per_class=20)
    
    if len(dataset) == 0:
        print("No data loaded!")
        return
    
    # Split dataset
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=dataset.labels[train_idx])
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=16, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=16, shuffle=False)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Test different configurations
    configs = [
        {'hidden_dim': 256, 'num_layers': 2, 'lr': 0.001},
        {'hidden_dim': 256, 'num_layers': 2, 'lr': 0.002},
        {'hidden_dim': 384, 'num_layers': 2, 'lr': 0.001},
    ]
    
    best_config = None
    best_acc = 0
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Config {i+1}: {config} ===")
        
        model = LSTMModel(
            input_dim=1659, 
            hidden_dim=config['hidden_dim'], 
            num_classes=num_classes, 
            num_layers=config['num_layers']
        )
        
        val_acc = train_quick_test(model, train_loader, val_loader, device, lr=config['lr'])
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config
            print(f"[NEW BEST] Config {i+1} achieved {val_acc:.4f} validation accuracy")
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    if best_acc > 0.1:  # If we get reasonable performance (>10% on 20 classes)
        print("✅ Training is working! The model can learn.")
    else:
        print("❌ Training may have issues. Very low accuracy.")

if __name__ == "__main__":
    main()
