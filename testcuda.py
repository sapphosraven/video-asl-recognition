"""
Script to verify CUDA availability for GPU acceleration.
This helps ensure that the system is properly configured for deep learning tasks.
"""

import torch
import sys

def check_cuda():
    """Check if CUDA is available and print device information."""
    print("PyTorch version:", torch.__version__)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU only.")
        return False
    
    print("CUDA is available! Details:")
    print(f"CUDA version: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"Number of available GPU devices: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {device_props.total_memory / (1024**3):.2f} GB")
        print(f"  CUDA capability: {device_props.major}.{device_props.minor}")
    
    # Set default device to GPU 0
    torch.cuda.set_device(0)
    print(f"Default device set to: {torch.cuda.get_device_name(0)}")
    
    # Quick tensor test
    print("\nRunning quick CUDA test...")
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    print(f"Test tensor sum on GPU: {a + b}")
    
    return True

if __name__ == "__main__":
    print("CUDA Availability Test")
    print("-" * 50)
    cuda_available = check_cuda()
    
    if cuda_available:
        print("\n✅ Your system is ready for GPU-accelerated deep learning!")
    else:
        print("\n⚠️ Your system will use CPU only, which may be slower for training.")
        print("Consider setting up CUDA if GPU acceleration is needed.")
    
    sys.exit(0 if cuda_available else 1)
