import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBlock(nn.Module):
    """Custom block combining ideas from VGG and ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(CustomBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv if needed
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # Squeeze-and-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = x
        
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        # Skip connection
        if self.skip is not None:
            identity = self.skip(x)
        
        # Combine and apply final activation
        out += identity
        out = F.relu(out)
        out = self.dropout(out)
        
        return out

class CustomCNN(nn.Module):
    """
    Custom CNN architecture combining ideas from LeNet, VGG, and ResNet.
    Features:
    - Residual connections
    - Batch normalization
    - Squeeze-and-Excitation attention
    - Dropout for regularization
    """
    def __init__(self, num_channels=3, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Custom blocks with increasing channels
        self.block1 = CustomBlock(32, 64, stride=2)
        self.block2 = CustomBlock(64, 128, stride=2)
        self.block3 = CustomBlock(128, 256, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Custom blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 