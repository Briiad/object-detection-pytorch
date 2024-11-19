import torch.nn as nn
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large as MobileNetV3

class CustomModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Load pretrained MobileNetV3 backbone
        mobilenet = MobileNetV3(
          weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        
        self.model = mobilenet

        # Print the number of parameters
        self.print_num_parameters()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total number of parameters: {total_params}')
