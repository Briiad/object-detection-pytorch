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
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {2, 4, 12}:  # Example indices
                features.append(x)
        print(f"Feature map shapes: {[f.shape for f in features]}")
        return features