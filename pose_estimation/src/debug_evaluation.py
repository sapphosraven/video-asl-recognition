#!/usr/bin/env python3
"""
Debug script to investigate evaluation issues
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add current dir to path
sys.path.append(os.path.dirname(__file__))

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

def debug_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load same data as training script
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
    
    # Split exactly like the main script
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42, stratify=dataset.labels[train_idx])
    
    # Load model
    config_path = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models\best_lstm_config.json'
    model_path = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models\best_lstm.pth'
    
    with open(config_path, 'r') as f:
        best_config = json.load(f)
    
    best_model = LSTMModel(input_dim=1659, hidden_dim=best_config['hidden_dim'], 
                          num_classes=num_classes, num_layers=best_config['num_layers'])
    best_model.dropout.p = best_config['dropout']
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.to(device)
    best_model.eval()
    
    # Test on a small subset of test data
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx[:32]), batch_size=8, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print(f"\nTesting on {len(test_idx[:32])} samples...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = best_model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            print(f"Batch {batch_idx + 1}:")
            for i in range(len(targets)):
                target_class = targets[i].item()
                pred_class = preds[i].item()
                max_prob = probs[i].max().item()
                target_prob = probs[i, target_class].item()
                
                print(f"  Sample {i}: Target={target_class} ({selected_classes[target_class]}), "
                      f"Pred={pred_class} ({selected_classes[pred_class]}), "
                      f"MaxProb={max_prob:.4f}, TargetProb={target_prob:.4f}")
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = (all_preds == all_targets).mean()
    print(f"\nAccuracy on test subset: {accuracy:.4f}")
    
    # Check if model is just predicting a few classes
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    print(f"\nPredicted classes distribution:")
    for pred_class, count in zip(unique_preds, pred_counts):
        print(f"  Class {pred_class} ({selected_classes[pred_class]}): {count} predictions")
    
    # Check if targets are distributed properly
    unique_targets, target_counts = np.unique(all_targets, return_counts=True)
    print(f"\nTarget classes distribution:")
    for target_class, count in zip(unique_targets, target_counts):
        print(f"  Class {target_class} ({selected_classes[target_class]}): {count} targets")

if __name__ == "__main__":
    debug_evaluation()
