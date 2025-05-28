"""
Quick test of the memory-efficient keypoint extraction on a few samples.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add the current directory to path to import the fixed extraction
sys.path.append(str(Path(__file__).parent))

from pose_estimation_fixed import process_single_instance

def test_extraction():
    """Test the fixed extraction on a few sample instances"""
    
    print("🧪 Testing Memory-Efficient Keypoint Extraction")
    print("=" * 50)
    
    data_dir = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\processed')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Find a few test instances
    test_instances = []
    for label_dir in list(data_dir.iterdir())[:3]:  # Test 3 labels
        if label_dir.is_dir():
            for instance_dir in list(label_dir.iterdir())[:2]:  # 2 instances per label
                if instance_dir.is_dir():
                    test_instances.append((instance_dir, label_dir.name))
    
    if not test_instances:
        print("❌ No test instances found!")
        return False
    
    print(f"📁 Testing {len(test_instances)} sample instances...")
    
    success_count = 0
    
    for i, (instance_dir, label) in enumerate(test_instances):
        print(f"\n{i+1}. Testing {label}/{instance_dir.name}")
        
        try:
            success = process_single_instance(instance_dir, label)
            
            if success:
                success_count += 1
                
                # Verify the output file
                output_dir = Path(r'f:\Uni_Stuff\6th_Sem\DL\Proj\video-asl-recognition\pose_estimation\data\keypoints')
                output_path = output_dir / label / f"{instance_dir.name}_keypoints.npz"
                
                if output_path.exists():
                    # Check the format
                    data = np.load(str(output_path))
                    if 'nodes' in data:
                        nodes = data['nodes']
                        print(f"   ✅ Output verified: {nodes.shape} (should be [frames, 553, 3])")
                        
                        if len(nodes.shape) == 3 and nodes.shape[1] == 553 and nodes.shape[2] == 3:
                            print(f"   🎯 Perfect format!")
                        else:
                            print(f"   ⚠️  Unexpected format: {nodes.shape}")
                    else:
                        print(f"   ❌ No 'nodes' key in output file")
                else:
                    print(f"   ❌ Output file not created")
            else:
                print(f"   ❌ Processing failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n📊 TEST RESULTS:")
    print("=" * 50)
    print(f"✅ Successful: {success_count}/{len(test_instances)}")
    print(f"📊 Success rate: {(success_count / len(test_instances) * 100):.1f}%")
    
    if success_count >= len(test_instances) * 0.8:  # 80% success rate
        print(f"\n🎉 TEST PASSED! Ready for full extraction.")
        print(f"💡 Run: python pose_estimation_fixed.py")
        return True
    else:
        print(f"\n⚠️  TEST ISSUES DETECTED!")
        print(f"💡 Check MediaPipe settings or video quality")
        return False

if __name__ == "__main__":
    test_extraction()
