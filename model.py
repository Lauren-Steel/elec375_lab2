import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()
        # Load ResNet18 backbone
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the fully connected layer from ResNet
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # Adaptive pooling to maintain spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer to regress to two outputs x, y coordinates of the snoot
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        # Pass input through ResNet
        features = self.resnet(x)
        # Apply adaptive pooling
        pooled_features = self.adaptive_pool(features)
        # Flatten pooled features
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # Get x and y coordinates
        keypoints = self.fc(pooled_features)
        return keypoints

# Testing model w a dummy input tensor 1 batch 3 channels 227x227 img
if __name__ == "__main__":
    model = SnoutNet()
    dummy_input = torch.randn(1, 3, 227, 227)
    output = model(dummy_input)

    # Test [1, 2]
    print(f"Output shape: {output.shape}")  

