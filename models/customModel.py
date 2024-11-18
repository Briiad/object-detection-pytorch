import torchvision
import torch.nn as nn
import torchvision.ops.FeaturePyramidNetwork as FPN
import torchvision.models.mobilenetv3.MobileNetV3 as MobileNetV3

from torchvision.models.detection.ssd import (
  SSD, 
  DefaultBoxGenerator,
  SSDHead
)

# MODEL FOR SSD WITH MOBILENETV3 BACKBONE AND FPN HEAD
class CustomModel(nn.Module):
  def __init__(self, num_classes=91):
    super().__init__()
    
    # Create MobileNetV3 backbone
    backbone = MobileNetV3(
      arch='large',
      weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,
      reduced_tail=False
    ).features
    
    # Define output channels for FPN
    return_layers = {
      '6': '0',    # stride 8
      '12': '1',   # stride 16
      '16': '2'    # stride 32
    }
    
    # Create Feature Pyramid Network
    in_channels = 960
    out_channels = 256
    
    self.backbone = torchvision.ops._utils.IntermediateLayerGetter(
      backbone, return_layers)
    self.fpn = FPN([in_channels] * 3, out_channels)
    
    # Create SSD head
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)])
    head = SSDHead(out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
    
    # Combine into SSD
    self.model = SSD(
      backbone=self.backbone,
      neck=self.fpn,
      head=head,
      anchor_generator=anchor_generator
    )
    
    # Print total parameters
    total_params = sum(p.numel() for p in self.parameters())
    print(f'Total parameters: {total_params:,}')
  
  def forward(self, x):
    return self.model(x)