import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

print("Testing PyTorch Geometric for ASL pose estimation...")

class PoseGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PoseGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

# Create a model suitable for pose estimation
model = PoseGNN(input_dim=2, hidden_dim=64, output_dim=10)
print(f"Model created successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Create sample pose data (simulating pose keypoints)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 0, 4],  # From nodes
    [1, 0, 2, 1, 3, 2, 4, 0]   # To nodes
], dtype=torch.long)

x = torch.tensor([
    [1.0, 2.0],  # pose keypoint 1
    [3.0, 4.0],  # pose keypoint 2  
    [5.0, 6.0],  # pose keypoint 3
    [7.0, 8.0],  # pose keypoint 4
    [9.0, 10.0]  # pose keypoint 5
], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
batch = Batch.from_data_list([data])

# Test forward pass
with torch.no_grad():
    output = model(batch)
    print(f"Forward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        batch_cuda = batch.cuda()
        output_cuda = model_cuda(batch_cuda)
        print(f"CUDA forward pass successful!")
        print(f"CUDA output shape: {output_cuda.shape}")

print("\n✅ PyTorch Geometric is working perfectly for your ASL pose estimation project!")
print("✅ You can now use Graph Neural Networks for pose-based sign language recognition!")
