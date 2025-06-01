#!/usr/bin/env python3
"""
Debug script to investigate model performance issues
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(__file__))

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

def debug_model():
    # Load model and config
    model_path = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models\best_lstm.pth'
    config_path = r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\models\best_lstm_config.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    print('Config:', config)

    # Create model
    model = LSTMModel(input_dim=1659, hidden_dim=config['hidden_dim'], 
                      num_classes=300, num_layers=config['num_layers'])

    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f'Model loaded on {device}')
    print(f'Model classifier shape: {model.classifier.weight.shape}')

    # Create dummy input to test model output
    dummy_input = torch.randn(4, 50, 1659).to(device)  # Batch of 4, sequence length 50, features 1659
    with torch.no_grad():
        output = model(dummy_input)
        print(f'Output shape: {output.shape}')
        print(f'Output sample logits (first sample):')
        print(f'  Min: {output[0].min().item():.3f}')
        print(f'  Max: {output[0].max().item():.3f}')
        print(f'  Mean: {output[0].mean().item():.3f}')
        print(f'  Std: {output[0].std().item():.3f}')
        
        # Check if model is making meaningful predictions
        probs = torch.softmax(output, dim=1)
        print(f'Probability distribution (first sample):')
        print(f'  Min prob: {probs[0].min().item():.6f}')
        print(f'  Max prob: {probs[0].max().item():.6f}')
        print(f'  Entropy: {-(probs[0] * torch.log(probs[0] + 1e-8)).sum().item():.3f}')
        print(f'  Expected entropy for random: {np.log(300):.3f}')
        
        # Check top predictions
        top_probs, top_indices = torch.topk(probs[0], 5)
        print(f'Top 5 predictions:')
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f'  {i+1}. Class {idx.item()}: {prob.item():.6f}')

if __name__ == "__main__":
    debug_model()
