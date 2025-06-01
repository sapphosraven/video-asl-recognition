"""
Configuration management and utility functions for the ASL recognition system.
"""
import json
import os
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the ASL recognition system"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "model_config": {
                "pose_model": {
                    "name": "tgcn",
                    "num_joints": 25,
                    "in_channels": 3,
                    "checkpoint_path": "OpenHands/checkpoints/model_tgcn_wlasl300.pth",
                    "target_sequence_length": 60,
                    "min_confidence": 0.25
                },
                "cnn_model": {
                    "checkpoint_path": "wordlevelrecogntion/asl_recognition_final_20250518_132050.pth",
                    "torchscript_path": "wordlevelrecogntion/asl_torchscript_20250518_132050.pt"
                }
            },
            "dataset_config": {
                "name": "WLASL300",
                "num_classes": 300,
                "class_mapping": "wordlevelrecogntion/class_map_wlasl300.json",
                "fallback_mapping": "wordlevelrecogntion/class_map.json"
            },
            "device": {
                "auto_detect": True,
                "fallback": "cpu"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for model inference"""
        if self.get('device.auto_detect', True):
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Using CUDA device for inference")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU device for inference")
        else:
            device_name = self.get('device.fallback', 'cpu')
            device = torch.device(device_name)
            logger.info(f"Using configured device: {device_name}")
        
        return device
    
    def get_pose_model_config(self) -> Dict[str, Any]:
        """Get pose model configuration"""
        return self.get('model_config.pose_model', {})
    
    def get_cnn_model_config(self) -> Dict[str, Any]:
        """Get CNN model configuration"""
        return self.get('model_config.cnn_model', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self.get('dataset_config', {})

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {}
    
    # Check OpenHands
    try:
        import openhands
        dependencies['openhands'] = True
        logger.info("OpenHands is available")
    except ImportError:
        dependencies['openhands'] = False
        logger.warning("OpenHands not found. Install with: pip install git+https://github.com/AI4Bharat/OpenHands.git")
    
    # Check torch-geometric
    try:
        import torch_geometric
        dependencies['torch_geometric'] = True
        logger.info("PyTorch Geometric is available")
    except ImportError:
        dependencies['torch_geometric'] = False
        logger.warning("PyTorch Geometric not found. Install with: pip install torch-geometric")
    
    # Check MediaPipe
    try:
        import mediapipe
        dependencies['mediapipe'] = True
        logger.info("MediaPipe is available")
    except ImportError:
        dependencies['mediapipe'] = False
        logger.warning("MediaPipe not found. Install with: pip install mediapipe")
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = True
        logger.info(f"PyTorch {torch.__version__} is available")
        if torch.cuda.is_available():
            logger.info(f"CUDA {torch.version.cuda} is available")
        else:
            logger.info("CUDA not available, using CPU")
    except ImportError:
        dependencies['torch'] = False
        logger.error("PyTorch not found. Install with: pip install torch torchvision")
    
    return dependencies

def verify_model_files(config: Config) -> Dict[str, bool]:
    """Verify that required model files exist"""
    files = {}
    
    # Check pose model checkpoint
    pose_checkpoint = config.get('model_config.pose_model.checkpoint_path')
    if pose_checkpoint:
        files['pose_model'] = os.path.exists(pose_checkpoint)
        if not files['pose_model']:
            logger.warning(f"Pose model checkpoint not found: {pose_checkpoint}")
    
    # Check CNN model files
    cnn_checkpoint = config.get('model_config.cnn_model.checkpoint_path')
    if cnn_checkpoint:
        files['cnn_model'] = os.path.exists(cnn_checkpoint)
        if not files['cnn_model']:
            logger.warning(f"CNN model checkpoint not found: {cnn_checkpoint}")
    
    torchscript_path = config.get('model_config.cnn_model.torchscript_path')
    if torchscript_path:
        files['cnn_torchscript'] = os.path.exists(torchscript_path)
    
    # Check class mapping files
    class_mapping = config.get('dataset_config.class_mapping')
    if class_mapping:
        files['class_mapping'] = os.path.exists(class_mapping)
        if not files['class_mapping']:
            logger.warning(f"Class mapping not found: {class_mapping}")
    
    fallback_mapping = config.get('dataset_config.fallback_mapping')
    if fallback_mapping:
        files['fallback_mapping'] = os.path.exists(fallback_mapping)
    
    return files

def system_check() -> bool:
    """Perform a comprehensive system check"""
    logger.info("Performing system check...")
    
    config = Config()
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [name for name, available in deps.items() if not available]
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {missing_deps}")
    
    # Check model files
    files = verify_model_files(config)
    missing_files = [name for name, exists in files.items() if not exists]
    
    if missing_files:
        logger.warning(f"Missing model files: {missing_files}")
    
    # Check critical components
    critical_components = ['torch', 'mediapipe']
    critical_missing = [dep for dep in critical_components if not deps.get(dep, False)]
    
    if critical_missing:
        logger.error(f"Critical dependencies missing: {critical_missing}")
        return False
    
    logger.info("System check completed")
    return True

# Global configuration instance
config = Config()
