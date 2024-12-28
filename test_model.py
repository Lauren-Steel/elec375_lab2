import torch
from model import SnoutNet  

# Create the model
model = SnoutNet()

# Create a random input tensor 
input_tensor = torch.randn(1, 3, 227, 227)

# Forward the input through the model
output = model(input_tensor)

# Check the output shape [1, 2]
print(output.shape)  