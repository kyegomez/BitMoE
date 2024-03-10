import torch
from bitmoe.main import BitMoE

# Set the parameters
dim = 10
hidden_dim = 20
output_dim = 30
num_experts = 5

# Create the model
model = BitMoE(dim, hidden_dim, output_dim, num_experts)

# Create random inputs
batch_size = 32
sequence_length = 100
x = torch.randn(batch_size, sequence_length, dim)

# Forward pass
output = model(x)

# Print the output shape
print(output)
print(output.shape)
