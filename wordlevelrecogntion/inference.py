import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from pathlib import Path
import json
import os

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
    """Load video, sample frames, resize, normalize for CNN input. Use middle frame by default."""
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total-1), num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        frames.append(img)
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {clip_path}")
    x = np.stack(frames, axis=0)  # (T, H, W, C)
    x = x.astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet stats
    x = torch.tensor(x).permute(0,3,1,2)  # (T, C, H, W)
    if use_middle_frame:
        # Use the middle frame only
        mid = x.shape[0] // 2
        x = x[mid:mid+1]
    else:
        # Average pool over time (legacy behavior)
        x = x.mean(dim=0, keepdim=True)
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

def predict_word_from_clip(model, clip_path, idx_to_class=idx_to_class, min_confidence=0.4, use_middle_frame=True):
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
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        x = preprocess_clip(clip_path, use_middle_frame=use_middle_frame)  # (1, C, H, W)
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        
        with torch.no_grad():
            # Forward pass
            logits = model(x.to(device))
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Check if model is making meaningful predictions
            prob_std = probs.std().item()
            if prob_std < 0.01:
                logger.warning(f"Model not discriminating well for {clip_path} (std={prob_std:.5f})")
                return "unknown"
            
            # Get prediction and confidence
            confidence, pred_idx = torch.max(probs, dim=0)
            pred = pred_idx.item()
            
            # Apply confidence threshold
            if confidence.item() < min_confidence:
                logger.warning(f"Low confidence prediction ({confidence.item()*100:.2f}%) for {clip_path}")
                return "unknown"
                
            # Map index to class name
            if idx_to_class is not None:
                # idx_to_class may have int keys or str keys
                return idx_to_class.get(str(pred), idx_to_class.get(pred, "unknown"))
            return str(pred)
            
    except Exception as e:
        logger.error(f"Error predicting word from clip {clip_path}: {str(e)}")
        return "unknown"
