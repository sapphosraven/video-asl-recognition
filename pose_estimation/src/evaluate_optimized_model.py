#!/usr/bin/env python3
"""
Evaluate the trained LSTM model with best hyperparameters
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
        lstm_out, (hidden, cell) = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        final_output = self.dropout(final_output)
        output = self.classifier(final_output)
        return output

def load_model_and_config(model_name='best_lstm_optimized'):
    """Load the trained model and its configuration"""
    models_dir = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models'
    
    model_path = os.path.join(models_dir, f'{model_name}.pth')
    config_path = os.path.join(models_dir, f'{model_name}_config.json')
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Model files not found: {model_path} or {config_path}")
        return None, None
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = LSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Model configuration: {config}")
    
    return model, config

def load_test_data():
    """Load test data for evaluation"""
    data_dir = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed'
    
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Test data files not found. Please run preprocessing first.")
        return None, None
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    # For evaluation, we'll use a subset or the full dataset
    # You might want to split this to match your training split
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split to get test data (same split as training)
    _, X_test, _, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    return X_test, y_test, label_encoder

def evaluate_model():
    """Evaluate the trained model"""
    print("Loading model and configuration...")
    model, config = load_model_and_config()
    
    if model is None:
        print("Failed to load model. Please train the model first.")
        return
    
    print("Loading test data...")
    X_test, y_test, label_encoder = load_test_data()
    
    if X_test is None:
        print("Failed to load test data.")
        return
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_test))}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate model
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), batch_size), desc="Evaluating"):
            batch_features = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            batch_labels = y_test[i:i+batch_size]
            
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n=== Model Evaluation Results ===")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    report = classification_report(all_labels, all_predictions, output_dict=True)
    print(f"Macro Average Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Average Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    # Analyze prediction confidence
    all_probabilities = np.array(all_probabilities)
    max_probs = np.max(all_probabilities, axis=1)
    
    print(f"\n=== Prediction Confidence Analysis ===")
    print(f"Mean prediction confidence: {np.mean(max_probs):.4f}")
    print(f"Median prediction confidence: {np.median(max_probs):.4f}")
    print(f"Min prediction confidence: {np.min(max_probs):.4f}")
    print(f"Max prediction confidence: {np.max(max_probs):.4f}")
    
    # Confidence distribution
    high_conf_mask = max_probs > 0.8
    medium_conf_mask = (max_probs > 0.5) & (max_probs <= 0.8)
    low_conf_mask = max_probs <= 0.5
    
    print(f"High confidence predictions (>0.8): {np.sum(high_conf_mask)} ({np.mean(high_conf_mask)*100:.1f}%)")
    print(f"Medium confidence predictions (0.5-0.8): {np.sum(medium_conf_mask)} ({np.mean(medium_conf_mask)*100:.1f}%)")
    print(f"Low confidence predictions (<=0.5): {np.sum(low_conf_mask)} ({np.mean(low_conf_mask)*100:.1f}%)")
    
    # Accuracy by confidence level
    if np.sum(high_conf_mask) > 0:
        high_conf_accuracy = accuracy_score(
            np.array(all_labels)[high_conf_mask], 
            np.array(all_predictions)[high_conf_mask]
        )
        print(f"Accuracy on high confidence predictions: {high_conf_accuracy:.4f}")
    
    # Top-k accuracy
    for k in [1, 3, 5]:
        if k <= len(np.unique(all_labels)):
            top_k_preds = np.argsort(all_probabilities, axis=1)[:, -k:]
            top_k_accuracy = np.mean([label in pred_top_k for label, pred_top_k in zip(all_labels, top_k_preds)])
            print(f"Top-{k} Accuracy: {top_k_accuracy:.4f} ({top_k_accuracy*100:.2f}%)")
    
    # Save detailed results
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confidence_stats': {
            'mean': float(np.mean(max_probs)),
            'median': float(np.median(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs))
        },
        'model_config': config
    }
    
    results_path = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models\evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create confusion matrix visualization (for smaller number of classes)
    if len(np.unique(all_labels)) <= 50:  # Only for manageable number of classes
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(all_labels, all_predictions)
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models\confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    evaluate_model()
