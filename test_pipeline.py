"""
Test script for the full ASL recognition pipeline with improved word recognition
"""
import os
import sys
import logging
from pipeline import process_asl_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_on_video(video_path):
    """Run the full pipeline on a video and display results"""
    print(f"\n{'='*50}")
    print(f"TESTING PIPELINE ON: {os.path.basename(video_path)}")
    print(f"{'='*50}")
    
    # Process the video
    results = process_asl_video(video_path)
    
    # Display results
    print(f"\nFOUND {len(results['clips'])} SEGMENTS:")
    for i, (clip, word) in enumerate(zip(results['clips'], results['words'])):
        clip_name = os.path.basename(clip)
        print(f"  {i+1}. {clip_name} → '{word}'")
    
    print(f"\nSENTENCE: {results['sentence']}")
    print(f"\nTIMING: {results['timing']['total']:.2f}s total")
    print(f"  - Segmentation: {results['timing']['segmentation']:.2f}s")
    print(f"  - Word recognition: {results['timing']['recognition']:.2f}s")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <path_to_video> [path_to_another_video] ...")
        sys.exit(1)
    
    # Test on all videos provided
    for video_path in sys.argv[1:]:
        test_on_video(video_path)
