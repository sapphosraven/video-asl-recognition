"""
Evaluation script for the best trained LSTM model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

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
            
            for filename in class_files:
                filepath = os.path.join(class_dir, filename)
                try:
                    data = np.load(filepath, allow_pickle=True)
                    # Use 'nodes' instead of 'keypoints' based on previous findings
                    keypoints = data['nodes'] if 'nodes' in data else data['keypoints']
                    
                    # Ensure we have the right shape and pad/truncate as needed
                    if len(keypoints) < max_sequence_length:
                        padding = np.zeros((max_sequence_length - len(keypoints), keypoints.shape[1]))
                        keypoints = np.vstack([keypoints, padding])
                    else:
                        keypoints = keypoints[:max_sequence_length]
                    
                    self.data.append(keypoints)
                    self.labels.append(self.class_to_idx[class_name])
                    
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue
        
        print(f"Loaded {len(self.data)} samples")
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last time step
        output = self.fc(lstm_out)
        return output

def evaluate_model():
    # Load configuration
    config_path = "f:/Uni_Stuff/6th_Sem/DL/Proj/video-asl-recognition/pose_estimation/models/best_lstm_config.json"
    model_path = "f:/Uni_Stuff/6th_Sem/DL/Proj/video-asl-recognition/pose_estimation/models/best_lstm.pth"
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print("Best model or config not found!")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Best model configuration: {config}")
      # Load data
    data_dir = "f:/Uni_Stuff/6th_Sem/DL/Proj/video-asl-recognition/pose_estimation/data/keypoints"
    wlasl_file = "f:/Uni_Stuff/6th_Sem/DL/Proj/video-asl-recognition/pose_estimation/data/WLASL300.json"
    
    with open(wlasl_file, 'r') as f:
        wlasl_data = json.load(f)
    
    # Extract class names from the WLASL data structure
    selected_classes = [item['gloss'] for item in wlasl_data][:300]  # All 300 classes
    print(f"Evaluating on {len(selected_classes)} classes")
    
    # Create dataset
    dataset = TestDataset(data_dir, selected_classes, max_samples_per_class=5)  # Small subset for quick eval
    
    if len(dataset) == 0:
        print("No data loaded!")
        return
    
    # Split into train/test for evaluation
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42, 
        stratify=dataset.labels
    )
    
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_size = dataset.data.shape[2]  # Feature dimension
    model = LSTMModel(
        input_size=input_size,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=len(selected_classes),
        dropout=config['dropout']
    )
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate
    all_predictions = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions: {total_correct}/{total_samples}")
    
    # Show per-class performance for top classes
    class_names = [selected_classes[i] for i in range(min(20, len(selected_classes)))]
    class_indices = list(range(min(20, len(selected_classes))))
    
    # Filter predictions and labels for top 20 classes only
    filtered_predictions = []
    filtered_labels = []
    for pred, label in zip(all_predictions, all_labels):
        if label in class_indices:
            filtered_predictions.append(pred)
            filtered_labels.append(label)
    
    if filtered_predictions:
        print(f"\nClassification Report (Top 20 classes, {len(filtered_predictions)} samples):")
        print(classification_report(filtered_labels, filtered_predictions, 
                                  target_names=class_names, zero_division=0))
    
    return accuracy, config

if __name__ == "__main__":
    evaluate_model()
