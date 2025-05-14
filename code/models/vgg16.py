
import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    """
    VGG16 architecture for fashion classification based on pretrained torchvision model.
    
    This implementation uses the VGG16 architecture from torchvision.models and adapts it
    for flexible input channels and output classes.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int, optional): Number of input channels (1 for grayscale, 3 for RGB)
                                       Default is 3.
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(VGG16, self).__init__()
        
        # Load the pretrained VGG16 model from torchvision
        self.model = models.vgg16(pretrained=True)
        
        # Modify the first layer if input_channels is not 3 (RGB)
        if input_channels != 3:
            # Get the first convolutional layer's parameters
            original_conv = self.model.features[0]
            # Create a new conv layer with the desired number of input channels
            new_conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize the new layer with weights adapted from the original layer
            # For grayscale, we can average the RGB weights
            # USE CASE: Fashion-MNIST Dataset
            if input_channels == 1:
                # Average the weights across the RGB channels
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
                    if original_conv.bias is not None:
                        new_conv.bias = nn.Parameter(original_conv.bias.clone())
            
            # Replace the first layer in the model
            self.model.features[0] = new_conv
        
        # Modify the classifier for the desired number of classes
        # The original classifier's last layer has 1000 outputs (ImageNet classes)
        # Replace it with a new layer with the desired number of outputs
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """Forward pass through the VGG16 model."""
        return self.model(x)


def get_vgg16_model(num_classes=10, input_channels=3, pretrained=True):
    """
    Initialize the VGG16 model using torchvision with pretrained weights.
    
    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        pretrained (bool): Whether to use pretrained weights from ImageNet
                          Default is True.
    
    Returns:
        VGG16: The initialized model
    """
    model = VGG16(num_classes=num_classes, input_channels=input_channels)
    
    # If not using pretrained weights, reinitialize all weights
    if not pretrained:
        for m in model.modules():
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
    
    return model