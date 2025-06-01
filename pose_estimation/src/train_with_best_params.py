#!/usr/bin/env python3
"""
Train LSTM model with the best hyperparameters found from optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import time
from tqdm import tqdm
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class LSTMModel(nn.Module):
    """LSTM-based architecture with optimized hyperparameters"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last output
        final_output = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        final_output = self.dropout(final_output)
        output = self.classifier(final_output)
        
        return output

class SimpleASLDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]

class ASLDataset(Dataset):
    def __init__(self, data_dir, selected_classes=None, max_sequence_length=50, max_samples_per_class=None):
        self.data = []
        self.labels = []
        
        # Get all available classes if none specified
        if selected_classes is None:
            selected_classes = [d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))]
            selected_classes.sort()  # Sort for consistency
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        
        logger.info(f"Loading data for {len(selected_classes)} classes...")
        
        for class_name in selected_classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            if max_samples_per_class:
                npy_files = npy_files[:max_samples_per_class]
            
            logger.info(f"Loading {len(npy_files)} samples for class '{class_name}'")
            
            for npy_file in npy_files:
                file_path = os.path.join(class_dir, npy_file)
                try:
                    keypoints = np.load(file_path)
                    
                    # Pad or truncate to max_sequence_length
                    if len(keypoints) > max_sequence_length:
                        keypoints = keypoints[:max_sequence_length]
                    elif len(keypoints) < max_sequence_length:
                        # Pad with zeros
                        pad_length = max_sequence_length - len(keypoints)
                        keypoints = np.pad(keypoints, ((0, pad_length), (0, 0)), mode='constant')
                    
                    self.data.append(keypoints)
                    self.labels.append(self.class_to_idx[class_name])
                    
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        logger.info(f"Dataset loaded: {len(self.data)} samples")
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Number of classes: {len(selected_classes)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

def load_processed_data():
    """Load preprocessed keypoint data from class directories"""
    data_dir = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints'
    
    logger.info("Loading processed data from class directories...")
    
    # Create dataset with all available classes
    dataset = ASLDataset(data_dir, selected_classes=None, max_sequence_length=50)
    
    if len(dataset) == 0:
        logger.error("No data loaded. Please check data directory and files.")
        return None, None, None, None
    
    # Extract features and labels
    features = dataset.data
    labels = dataset.labels
    
    logger.info(f"Loaded {len(features)} samples")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Unique classes: {len(np.unique(labels))}")
    
    # Create label encoder for class names
    class_names = list(dataset.class_to_idx.keys())
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    return features, labels, label_encoder, dataset.class_to_idx

def train_model():
    """Train LSTM model with best hyperparameters"""
    
    # Best hyperparameters from optimization
    BEST_PARAMS = {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.3,
        'lr': 0.0005
    }
    
    logger.info(f"Training with best hyperparameters: {BEST_PARAMS}")    # Load data
    features, labels, label_encoder, class_to_idx = load_processed_data()
    if features is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Testing samples: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = SimpleASLDataset(X_train, y_train)
    test_dataset = SimpleASLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = features.shape[2]  # Feature dimension
    num_classes = len(np.unique(labels))
    
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=BEST_PARAMS['hidden_dim'],
        num_classes=num_classes,
        num_layers=BEST_PARAMS['num_layers'],
        dropout=BEST_PARAMS['dropout']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Model initialized on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    best_accuracy = 0.0
    patience = 20
    patience_counter = 0
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_features, batch_labels in progress_bar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        
        # Update learning rate scheduler
        scheduler.step(test_accuracy)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Train Acc: {train_accuracy:.2f}%, '
                   f'Test Acc: {test_accuracy:.2f}%')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
              # Save model and config
            models_dir = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models'
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, 'best_lstm_optimized.pth')
            config_path = os.path.join(models_dir, 'best_lstm_optimized_config.json')
            
            torch.save(model.state_dict(), model_path)
            
            config_data = {
                **BEST_PARAMS,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'best_accuracy': best_accuracy,
                'epoch': epoch + 1,
                'class_to_idx': class_to_idx
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {patience} epochs without improvement')
                break
    
    logger.info(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Generate classification report
    report = classification_report(all_labels, all_predictions, output_dict=True)
    
    # Save final results
    results = {
        'best_accuracy': best_accuracy,
        'final_accuracy': accuracy_score(all_labels, all_predictions) * 100,
        'hyperparameters': BEST_PARAMS,
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        },
        'classification_report': report
    }
    
    results_path = os.path.join(models_dir, 'training_results_optimized.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f'Training results saved to {results_path}')
    
    return model, label_encoder

if __name__ == "__main__":
    train_model()
