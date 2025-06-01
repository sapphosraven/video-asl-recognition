import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from pathlib import Path
import json
import os
from PIL import Image
import logging

# --- Model definition (should match the one used in training) ---
class ASLWordCNN(nn.Module):
    def __init__(self, num_classes=300):
        super().__init__()
        from torchvision.models import mobilenet_v2
        self.base = mobilenet_v2(pretrained=False)
        self.base.classifier[1] = nn.Linear(self.base.last_channel, num_classes)
    def forward(self, x):
        return self.base(x)

def load_cnn_model(model_path, num_classes=300, device=None):
    """Load the trained CNN model from a .pth file or checkpoint dict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLWordCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'base.features.0.0.weight' in checkpoint:
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model

# --- Preprocessing ---
def preprocess_clip(clip_path, num_frames=16, size=(240,240), use_middle_frame=True):
    """Load video, sample frames, resize via PIL, normalize for CNN input."""
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total-1), num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert BGR to RGB and use PIL for resizing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize(size, Image.BILINEAR)
        img = np.array(pil_img)
        frames.append(img)
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {clip_path}")
    x = np.stack(frames, axis=0)  # (T, H, W, C)
    x = x.astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.tensor(x).permute(0,3,1,2)  # (T, C, H, W)
    if use_middle_frame:
        mid = x.shape[0] // 2
        x = x[mid:mid+1]
    else:
        x = x  # keep all frames for ensemble
    return x.float()

# --- Inference ---
# Load class mapping from training notebook
# Try to load mapping from a file, fallback to None
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), 'class_map.json')
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        idx_to_class = json.load(f)
else:
    idx_to_class = None

def predict_word_from_clip(model, clip_path, idx_to_class=idx_to_class, min_confidence=0.25, use_middle_frame=True):
    """
    Predict the word label for a video clip with confidence check.
    
    Args:
        model: The trained ASL recognition model
        clip_path: Path to the video clip
        idx_to_class: Dictionary mapping class indices to word labels
        min_confidence: Minimum confidence threshold (default: 0.4)
        use_middle_frame: If True, use only the middle frame for prediction (recommended)
        
    Returns:
        String containing the predicted word or "unknown" if confidence is too low
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[DEBUG] Starting prediction for clip: {clip_path}")
    
    try:
        # Preprocess and inspect tensor
        x = preprocess_clip(clip_path, use_middle_frame=use_middle_frame)  # (1, C, H, W)
        logger.info(f"[DEBUG] Preprocessed tensor shape: {tuple(x.shape)}")
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        
        with torch.no_grad():
            # Forward pass
            logits = model(x.to(device))
            logger.info(f"[DEBUG] Logits: {logits.cpu().numpy()}")
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            logger.info(f"[DEBUG] Probabilities sum: {probs.sum().item():.4f}, std: {probs.std().item():.6f}")
            
            # Check if model is making meaningful predictions (disabled for demo)
            prob_std = probs.std().item()
            logger.info(f"[DEBUG] Discrimination std: {prob_std:.6f}")
             
            # Get prediction and confidence
            confidence, pred_idx = torch.max(probs, dim=0)
            logger.info(f"[DEBUG] Max confidence: {confidence.item()*100:.2f}%, Pred idx: {pred_idx.item()}")
            pred = pred_idx.item()
            
            # Apply confidence threshold
            if confidence.item() < min_confidence:
                logger.warning(f"[DEBUG] Confidence below threshold ({confidence.item():.2f}), returning 'unknown'")
                return "unknown"
            
            # Map index to class name
            if idx_to_class is not None:
                # idx_to_class may have int keys or str keys
                mapped = idx_to_class.get(str(pred), idx_to_class.get(pred, None))
                logger.info(f"[DEBUG] Mapped pred to word: {mapped}")
                return mapped or "unknown"
            return str(pred)
            
    except Exception as e:
        logger.error(f"Error predicting word from clip {clip_path}: {str(e)}")
        return "unknown"

def predict_word_from_clip_ensemble(model, clip_path, idx_to_class=idx_to_class, min_confidence=0.2, temperature=2.0):
    """Predict word by averaging logits over all frames with temperature scaling."""
    logger = logging.getLogger(__name__)
    try:
        # Preprocess to get all frames
        x = preprocess_clip(clip_path, use_middle_frame=False)  # (T, C, H, W)
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(x.to(device))  # shape (T, C)
            logits_mean = logits.mean(dim=0, keepdim=True)  # (1, C)
            
            # Apply temperature scaling to calibrate confidence
            scaled_logits = logits_mean / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)[0]
            
            # Log top predictions for debugging
            top_probs, top_indices = torch.topk(probs, 3)
            logger.info(f"Top predictions for {clip_path} (temp={temperature}):")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                word = idx_to_class.get(str(idx.item()), str(idx.item())) if idx_to_class else str(idx.item())
                logger.info(f"  {i+1}. {word}: {prob.item()*100:.2f}%")
            
            # Disabled discrimination std check for demo
            std_val = probs.std().item()
            logger.info(f"[DEBUG] Ensemble discrimination std: {std_val:.6f}")
            confidence, pred_idx = torch.max(probs, dim=0)
            if confidence.item() < min_confidence:
                logger.warning(f"Low confidence {confidence.item()*100:.2f}% for {clip_path}")
                return "unknown"
            pred = pred_idx.item()
            if idx_to_class:
                predicted_word = idx_to_class.get(str(pred), idx_to_class.get(pred, "unknown"))
                logger.info(f"Final prediction for {clip_path}: {predicted_word} ({confidence.item()*100:.2f}%)")
                return predicted_word
            return str(pred)
    except Exception as e:
        logger.error(f"Error ensemble predicting {clip_path}: {e}")
        return "unknown"
