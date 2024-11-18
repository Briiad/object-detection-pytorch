import torch
from torch import nn
from .backbone import MobileNetV3Backbone
from .fpnlite import FPNLite
from .ssd import SSDHead

class DetectionModel(nn.Module):
    """
    Example:
    model = DetectionModel(num_classes=20)
    """
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.backbone = MobileNetV3Backbone()
        self.fpn = FPNLite([24, 40, 112], 256)  # Channel numbers based on backbone layer outputs
        self.head = SSDHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        cls_preds, box_preds = self.head(fpn_features)
        return cls_preds, box_preds
