"""
CNN Model Testing Script
Comprehensive testing of the ASL word-level CNN model similar to original notebook testing.

This script tests:
1. Model loading and architecture
2. Preprocessing pipeline
3. Prediction accuracy on sample videos
4. Class mapping consistency
5. Confidence calibration analysis
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Import the CNN model and inference functions
from wordlevelrecogntion.inference import (
    ASLWordCNN, 
    load_cnn_model, 
    predict_word_from_clip,
    predict_word_from_clip_ensemble,
    preprocess_clip
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CNNModelTester:
    """Comprehensive CNN model testing class."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "wordlevelrecogntion/asl_recognition_final_20250518_132050.pth"
        self.class_map_path = "wordlevelrecogntion/class_map_correct.json"
        self.uploads_dir = Path("uploads")
        
        self.model = None
        self.class_map = None
        self.test_results = {}
        
        print(f"🔧 Using device: {self.device}")
        print(f"📁 Model path: {self.model_path}")
        print(f"📁 Class map path: {self.class_map_path}")
        print(f"📁 Test videos directory: {self.uploads_dir}")
        
    def load_class_mapping(self) -> bool:
        """Load and validate class mapping."""
        print("\n" + "="*60)
        print("📋 LOADING CLASS MAPPING")
        print("="*60)
        
        try:
            if not os.path.exists(self.class_map_path):
                print(f"❌ Class mapping file not found: {self.class_map_path}")
                return False
                
            with open(self.class_map_path, 'r', encoding='utf-8') as f:
                self.class_map = json.load(f)
            
            print(f"✅ Loaded class mapping with {len(self.class_map)} classes")
            print(f"📊 Sample classes:")
            
            # Show first 10 classes
            for i, (idx, word) in enumerate(list(self.class_map.items())[:10]):
                print(f"   {idx}: {word}")
            if len(self.class_map) > 10:
                print(f"   ... and {len(self.class_map) - 10} more")
                
            # Validate mapping consistency
            expected_classes = 300
            if len(self.class_map) != expected_classes:
                print(f"⚠️  Warning: Expected {expected_classes} classes, got {len(self.class_map)}")
                
            # Check if indices are continuous
            indices = sorted([int(k) for k in self.class_map.keys()])
            if indices != list(range(len(indices))):
                print(f"⚠️  Warning: Class indices are not continuous!")
                print(f"   Range: {min(indices)} to {max(indices)}")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load class mapping: {e}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test CNN model loading and architecture."""
        print("\n" + "="*60)
        print("🤖 TESTING MODEL LOADING")
        print("="*60)
        
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
                
            print(f"📁 Loading model from: {self.model_path}")
            start_time = time.time()
            
            # Load the model
            self.model = load_cnn_model(
                self.model_path, 
                num_classes=300,
                device=self.device
            )
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            
            # Test model architecture
            print(f"\n🏗️  Model Architecture Analysis:")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Model device: {next(self.model.parameters()).device}")
            print(f"   Model type: {type(self.model).__name__}")
            
            # Test forward pass with dummy input
            print(f"\n🧪 Testing forward pass...")
            dummy_input = torch.randn(1, 3, 240, 240).to(self.device)
            
            with torch.no_grad():
                start_time = time.time()
                output = self.model(dummy_input)
                inference_time = time.time() - start_time
                
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Inference time: {inference_time*1000:.2f} ms")
            
            # Analyze output distribution
            output_np = output.cpu().numpy()[0]
            print(f"   Output stats: min={output_np.min():.3f}, max={output_np.max():.3f}, std={output_np.std():.3f}")
            
            # Test softmax probabilities
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            max_prob = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            
            print(f"   Max probability: {max_prob:.4f}")
            print(f"   Entropy: {entropy:.4f} (higher = more uncertain)")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def test_preprocessing_pipeline(self) -> bool:
        """Test video preprocessing pipeline."""
        print("\n" + "="*60)
        print("🎬 TESTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # Find a test video
        test_videos = list(self.uploads_dir.glob("*.mp4"))[:3]  # Test first 3 videos
        
        if not test_videos:
            print(f"❌ No test videos found in {self.uploads_dir}")
            return False
            
        print(f"📁 Found {len(test_videos)} test videos")
        
        for video_path in test_videos:
            print(f"\n🎥 Testing: {video_path.name}")
            
            try:
                # Test basic video info
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"   ❌ Failed to open video")
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                print(f"   📊 Video info: {width}x{height}, {frame_count} frames, {fps:.1f} fps, {duration:.1f}s")
                cap.release()
                
                # Test preprocessing with different settings
                print(f"   🔄 Testing preprocessing...")
                
                # Test middle frame extraction
                start_time = time.time()
                frames_single = preprocess_clip(video_path, num_frames=16, use_middle_frame=True)
                single_time = time.time() - start_time
                
                print(f"   ✅ Single frame: {frames_single.shape} in {single_time*1000:.1f}ms")
                
                # Test multi-frame extraction
                start_time = time.time()
                frames_multi = preprocess_clip(video_path, num_frames=16, use_middle_frame=False)
                multi_time = time.time() - start_time
                
                print(f"   ✅ Multi frames: {frames_multi.shape} in {multi_time*1000:.1f}ms")
                
                # Analyze preprocessed data
                frames_np = frames_single.numpy()
                print(f"   📊 Frame stats: min={frames_np.min():.3f}, max={frames_np.max():.3f}, mean={frames_np.mean():.3f}")
                
                # Check if normalization looks correct (should be roughly [-2, 2] for ImageNet normalization)
                if frames_np.min() < -3 or frames_np.max() > 3:
                    print(f"   ⚠️  Warning: Frame values seem outside expected range")
                    
            except Exception as e:
                print(f"   ❌ Preprocessing failed: {e}")
                continue
                
        return True
    
    def test_predictions_on_samples(self) -> bool:
        """Test model predictions on sample videos."""
        print("\n" + "="*60)
        print("🎯 TESTING PREDICTIONS ON SAMPLE VIDEOS")
        print("="*60)
        
        if self.model is None:
            print("❌ Model not loaded")
            return False
            
        # Get some test videos - prefer clips that have specific word IDs
        test_videos = []
        
        # Look for videos that might correspond to known words
        known_word_videos = {
            "00414": "about",  # Video 414 should be "about" 
            "00623": "accident",
            "02999": "again", 
            "04694": "all",
            "05229": "always"
        }
        
        # Add specific videos if they exist
        for vid_id, expected_word in known_word_videos.items():
            video_path = self.uploads_dir / f"{vid_id}.mp4"
            if video_path.exists():
                test_videos.append((video_path, expected_word))
        
        # Add some random clips as well
        other_videos = [v for v in self.uploads_dir.glob("*.mp4") 
                       if not any(v.name.startswith(vid_id) for vid_id in known_word_videos)][:3]
        for video_path in other_videos:
            test_videos.append((video_path, "unknown"))
        
        if not test_videos:
            print("❌ No test videos available")
            return False
            
        print(f"🧪 Testing on {len(test_videos)} videos")
        
        results = []
        
        for i, (video_path, expected_word) in enumerate(test_videos):
            print(f"\n🎬 Test {i+1}: {video_path.name}")
            print(f"   Expected: {expected_word}")
            
            try:
                # Test single frame prediction
                start_time = time.time()
                pred_single = predict_word_from_clip(
                    self.model, 
                    video_path, 
                    idx_to_class=self.class_map,
                    min_confidence=0.1,  # Lower threshold for testing
                    use_middle_frame=True
                )
                single_time = time.time() - start_time
                
                print(f"   🎯 Single frame prediction: '{pred_single}' ({single_time*1000:.1f}ms)")
                
                # Test ensemble prediction
                start_time = time.time()
                pred_ensemble = predict_word_from_clip_ensemble(
                    self.model,
                    video_path,
                    idx_to_class=self.class_map,
                    min_confidence=0.1,
                    temperature=2.0
                )
                ensemble_time = time.time() - start_time
                
                print(f"   🎯 Ensemble prediction: '{pred_ensemble}' ({ensemble_time*1000:.1f}ms)")
                
                # Get detailed prediction analysis
                detailed_analysis = self._analyze_prediction_details(video_path)
                
                result = {
                    'video': video_path.name,
                    'expected': expected_word,
                    'pred_single': pred_single,
                    'pred_ensemble': pred_ensemble,
                    'single_time': single_time,
                    'ensemble_time': ensemble_time,
                    'details': detailed_analysis
                }
                
                results.append(result)
                
                # Check accuracy
                if expected_word != "unknown":
                    single_correct = pred_single.lower() == expected_word.lower()
                    ensemble_correct = pred_ensemble.lower() == expected_word.lower()
                    
                    print(f"   ✅ Single frame correct: {single_correct}")
                    print(f"   ✅ Ensemble correct: {ensemble_correct}")
                    
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                continue
        
        # Summary statistics
        print(f"\n📊 PREDICTION SUMMARY")
        print("="*40)
        
        self.test_results['predictions'] = results
        
        if results:
            avg_single_time = np.mean([r['single_time'] for r in results]) * 1000
            avg_ensemble_time = np.mean([r['ensemble_time'] for r in results]) * 1000
            
            print(f"Average inference time:")
            print(f"  Single frame: {avg_single_time:.1f}ms")
            print(f"  Ensemble: {avg_ensemble_time:.1f}ms")
            
            # Accuracy on known videos
            known_results = [r for r in results if r['expected'] != "unknown"]
            if known_results:
                single_acc = sum(1 for r in known_results 
                               if r['pred_single'].lower() == r['expected'].lower()) / len(known_results)
                ensemble_acc = sum(1 for r in known_results 
                                 if r['pred_ensemble'].lower() == r['expected'].lower()) / len(known_results)
                
                print(f"\nAccuracy on known videos ({len(known_results)} videos):")
                print(f"  Single frame: {single_acc*100:.1f}%")
                print(f"  Ensemble: {ensemble_acc*100:.1f}%")
        
        return True
    
    def _analyze_prediction_details(self, video_path: Path) -> Dict[str, Any]:
        """Get detailed prediction analysis for a video."""
        try:
            # Preprocess video
            frames = preprocess_clip(video_path, use_middle_frame=True)
            
            with torch.no_grad():
                # Get raw logits
                logits = self.model(frames.to(self.device))
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Get top 5 predictions
                top_probs, top_indices = torch.topk(probs, 5)
                
                top_predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    word = self.class_map.get(str(idx.item()), f"class_{idx.item()}")
                    top_predictions.append({
                        'word': word,
                        'confidence': prob.item(),
                        'class_idx': idx.item()
                    })
                
                # Calculate prediction statistics
                max_prob = probs.max().item()
                entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                prob_std = probs.std().item()
                
                return {
                    'top_predictions': top_predictions,
                    'max_confidence': max_prob,
                    'entropy': entropy,
                    'probability_std': prob_std,
                    'is_confident': max_prob > 0.3 and prob_std > 0.01
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def test_confidence_calibration(self) -> bool:
        """Test confidence calibration across different video types."""
        print("\n" + "="*60)
        print("📊 TESTING CONFIDENCE CALIBRATION")
        print("="*60)
        
        if not self.test_results.get('predictions'):
            print("❌ No prediction results available")
            return False
            
        results = self.test_results['predictions']
        
        # Analyze confidence distributions
        all_confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        for result in results:
            details = result.get('details', {})
            if 'top_predictions' in details and details['top_predictions']:
                confidence = details['top_predictions'][0]['confidence']
                all_confidences.append(confidence)
                
                if result['expected'] != "unknown":
                    is_correct = (result['pred_single'].lower() == result['expected'].lower())
                    if is_correct:
                        correct_confidences.append(confidence)
                    else:
                        incorrect_confidences.append(confidence)
        
        if all_confidences:
            print(f"📈 Confidence Statistics:")
            print(f"   All predictions: mean={np.mean(all_confidences):.3f}, std={np.std(all_confidences):.3f}")
            print(f"   Range: {np.min(all_confidences):.3f} to {np.max(all_confidences):.3f}")
            
            if correct_confidences:
                print(f"   Correct predictions: mean={np.mean(correct_confidences):.3f}")
            
            if incorrect_confidences:
                print(f"   Incorrect predictions: mean={np.mean(incorrect_confidences):.3f}")
            
            # Confidence thresholds analysis
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            print(f"\n🎯 Confidence Threshold Analysis:")
            
            for threshold in thresholds:
                high_conf_results = [r for r in results 
                                   if r.get('details', {}).get('top_predictions', [{}])[0].get('confidence', 0) >= threshold]
                
                coverage = len(high_conf_results) / len(results) * 100
                
                if high_conf_results:
                    known_high_conf = [r for r in high_conf_results if r['expected'] != "unknown"]
                    if known_high_conf:
                        accuracy = sum(1 for r in known_high_conf 
                                     if r['pred_single'].lower() == r['expected'].lower()) / len(known_high_conf) * 100
                        print(f"   Threshold {threshold:.1f}: {coverage:.1f}% coverage, {accuracy:.1f}% accuracy")
                    else:
                        print(f"   Threshold {threshold:.1f}: {coverage:.1f}% coverage, no known labels")
                else:
                    print(f"   Threshold {threshold:.1f}: 0% coverage")
        
        return True
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report_lines = []
        report_lines.append("ASL CNN MODEL TEST REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Device: {self.device}")
        report_lines.append("")
        
        # Model info
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            report_lines.append(f"Model: {type(self.model).__name__}")
            report_lines.append(f"Parameters: {total_params:,}")
            report_lines.append(f"Classes: {len(self.class_map) if self.class_map else 'Unknown'}")
            report_lines.append("")
        
        # Prediction results
        if self.test_results.get('predictions'):
            results = self.test_results['predictions']
            report_lines.append(f"PREDICTION RESULTS ({len(results)} videos tested)")
            report_lines.append("-" * 30)
            
            for result in results:
                report_lines.append(f"Video: {result['video']}")
                report_lines.append(f"  Expected: {result['expected']}")
                report_lines.append(f"  Predicted: {result['pred_single']}")
                
                details = result.get('details', {})
                if 'top_predictions' in details:
                    report_lines.append(f"  Top 3 predictions:")
                    for i, pred in enumerate(details['top_predictions'][:3]):
                        report_lines.append(f"    {i+1}. {pred['word']}: {pred['confidence']:.3f}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def run_all_tests(self) -> bool:
        """Run all CNN model tests."""
        print("🚀 STARTING COMPREHENSIVE CNN MODEL TESTING")
        print("=" * 60)
        
        success = True
        
        # Test 1: Load class mapping
        if not self.load_class_mapping():
            success = False
            
        # Test 2: Load model
        if not self.test_model_loading():
            success = False
            
        # Test 3: Test preprocessing
        if not self.test_preprocessing_pipeline():
            success = False
            
        # Test 4: Test predictions
        if not self.test_predictions_on_samples():
            success = False
            
        # Test 5: Test confidence calibration
        if not self.test_confidence_calibration():
            success = False
        
        # Generate report
        print("\n" + "="*60)
        print("📋 GENERATING TEST REPORT")
        print("="*60)
        
        report = self.generate_test_report()
        
        # Save report to file
        report_path = "cnn_test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Test report saved to: {report_path}")
        
        # Final summary
        print("\n" + "="*60)
        print("🏁 TESTING COMPLETE")
        print("="*60)
        
        if success:
            print("✅ All tests completed successfully!")
            print("🎯 Model appears to be working correctly")
            print("💡 Check the detailed report for performance analysis")
        else:
            print("❌ Some tests failed!")
            print("🔧 Check the error messages above for debugging")
            
        return success


def main():
    """Main testing function."""
    print("🧪 ASL CNN Model Testing Script")
    print("Testing CNN model similar to original notebook validation")
    print("=" * 60)
    
    # Create tester instance
    tester = CNNModelTester()
    
    # Run all tests
    success = tester.run_all_tests()
    
    # Print final status
    print(f"\n🎬 Testing {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    main()
