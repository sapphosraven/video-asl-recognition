import os
import sys
import cv2
import numpy as np
import torch
from wordlevelrecogntion.inference import preprocess_clip, load_cnn_model, predict_word_from_clip
import json
import matplotlib.pyplot as plt

def visualize_preprocessed_clip(clip_path, save_path=None):
    """Visualize what the preprocessing is doing to a clip"""
    print(f"Analyzing clip: {clip_path}")
    
    # Get basic video info
    cap = cv2.VideoCapture(clip_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video info: {frame_count} frames, {fps} FPS")
    
    # Process the clip
    try:
        x = preprocess_clip(clip_path)
        
        # Check tensor values
        print(f"Preprocessed tensor shape: {x.shape}")
        print(f"Tensor stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        # Visualize the preprocessed frame (averaged over time)
        if save_path:
            # Convert back to image format
            img = x[0].permute(1, 2, 0).numpy()
            # Denormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Preprocessed {os.path.basename(clip_path)}")
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        
        return True
    except Exception as e:
        print(f"Error preprocessing clip: {e}")
        return False

def debug_model(model_path, clip_path):
    """Debug the model's prediction process for a clip"""
    print(f"Loading model from {model_path}")
    
    # Try to load TorchScript model first
    torchscript_path = os.path.join(os.path.dirname(model_path), 'asl_torchscript_20250518_132050.pt')
    model = None
    
    try:
        if os.path.exists(torchscript_path):
            print(f"Attempting to load TorchScript model from {torchscript_path}")
            model = torch.jit.load(torchscript_path)
            print("Successfully loaded TorchScript model")
        else:
            print("TorchScript model not found, loading regular model")
            model = load_cnn_model(model_path)
            print("Successfully loaded regular PyTorch model")
    except Exception as e:
        print(f"Error loading model: {e}")
        if model is None:
            print("Attempting to load regular model as fallback")
            try:
                model = load_cnn_model(model_path)
            except Exception as e2:
                print(f"Failed to load regular model too: {e2}")
                return
    
    # Load class map
    class_map_path = os.path.join(os.path.dirname(model_path), 'class_map.json')
    if os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            idx_to_class = json.load(f)
            print(f"Loaded class map with {len(idx_to_class)} classes")
    else:
        print("Warning: class_map.json not found")
        idx_to_class = None
    
    # Preprocess the clip
    try:
        x = preprocess_clip(clip_path)
        print(f"Input shape: {x.shape}")
        
        # Check if model contains any NaN weights (potential issue)
        has_nan = False
        if hasattr(model, 'parameters'):
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"WARNING: NaN values found in model parameters: {name}")
                    has_nan = True
                    
        if has_nan:
            print("Model contains NaN weights which will result in incorrect predictions")
        
        # Run prediction with detailed debug info
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        print(f"Using device: {device}")
        
        with torch.no_grad():
            try:
                # Try to manually trace through model layers
                if hasattr(model, 'base') and hasattr(model.base, 'features'):
                    print("Checking feature extraction...")
                    features = model.base.features(x.to(device))
                    print(f"Feature shape: {features.shape}, range: [{features.min().item():.2f}, {features.max().item():.2f}]")
                    
                # Full forward pass
                print("Running full prediction...")
                logits = model(x.to(device))
                print(f"Logits shape: {logits.shape}, range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                
                # Detailed probabilities analysis
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                print(f"Probability range: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
                print(f"Standard deviation of probabilities: {probs.std().item():.6f}")
                
                # Histogram of probabilities (simplified)
                bins = [0, 0.001, 0.01, 0.1, 0.5, 1.0]
                counts = [(probs < bins[i+1]) & (probs >= bins[i]).sum().item() for i in range(len(bins)-1)]
                print("Probability distribution:")
                for i, count in enumerate(counts):
                    print(f"  {bins[i]:.3f} - {bins[i+1]:.3f}: {count} classes")
                
                # Print top 5 predictions
                top_probs, top_indices = torch.topk(probs, 5)
                
                print("\nTop 5 predictions:")
                for i, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
                    label = idx_to_class.get(str(idx), str(idx)) if idx_to_class else str(idx)
                    print(f"{i+1}. {label}: {prob*100:.2f}%")
                
                # Check for issues
                if probs.max().item() > 0.9:
                    print("\nWarning: Prediction is suspiciously confident (>90%)")
                
                if probs.std().item() < 0.01:
                    print("\nWarning: Probabilities have low standard deviation, model may not be discriminating well")
                    
                # Get behind probability (since that's what seems to be happening)
                behind_idx = 27  # index for "behind" from class_map.json
                if idx_to_class and '27' in idx_to_class:
                    behind_prob = probs[behind_idx].item()
                    print(f"\nProbability for 'behind' (idx 27): {behind_prob*100:.6f}%")
                    
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_recognition.py <clip_path> [model_path]")
        sys.exit(1)
        
    clip_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'wordlevelrecogntion/asl_recognition_final_20250518_132050.pth'
    
    print("=" * 50)
    print("ASL RECOGNITION DEBUGGING TOOL")
    print("=" * 50)
    
    # Visualize preprocessing
    visualize_preprocessed_clip(clip_path)
    
    # Debug model predictions
    print("\n" + "=" * 50)
    print("MODEL PREDICTION DEBUG")
    print("=" * 50)
    debug_model(model_path, clip_path)
    vis_path = f"debug_{os.path.basename(clip_path).replace('.mp4', '.png')}"
    visualize_preprocessed_clip(clip_path, vis_path)
    
    # Debug model prediction
    print("\n" + "=" * 50)
    debug_model(model_path, clip_path)
    print("=" * 50)
