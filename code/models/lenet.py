import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    LeNet-5 CNN architecture implementation.
    Architecture:
    - Conv1 (num_channels->6, 5x5) -> ReLU -> MaxPool2d
    - Conv2 (6->16, 5x5) -> ReLU -> MaxPool2d
    - Adaptive pooling to ensure fixed feature map size
    - FC1 (16*5*5->120) -> ReLU
    - FC2 (120->84) -> ReLU
    - FC3 (84->num_classes)
    """
    def __init__(self, num_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # Ensure 5x5 feature maps
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # First convolutional layer followed by ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Second convolutional layer followed by ReLU and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Adaptive pooling to ensure fixed feature map size
        x = self.adaptive_pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Three fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
