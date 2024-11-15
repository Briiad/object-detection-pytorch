import torch
import torchvision.models as models
from torch import nn

class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Backbone, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Use specific layers from MobileNetV3 for feature extraction
        self.features = mobilenet.features[:]

    def forward(self, x):
        return self.features(x)