import torch
from torch import nn
from .backbone import MobileNetV3Backbone
from .fpnlite import FPNLite
from .ssd import SSDHead

# In model.py, update DetectionModel class

class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.num_classes = num_classes  # Add this line
        self.backbone = MobileNetV3Backbone()
        self.fpn = FPNLite([24, 40, 112], 256)
        self.head = SSDHead(num_classes)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        
        # Get predictions from head
        cls_preds_list, box_preds_list = self.head(fpn_features)
        
        # Stack predictions if they're in list form
        if isinstance(cls_preds_list, list):
            cls_preds = torch.cat(cls_preds_list, dim=1)
        else:
            cls_preds = cls_preds_list
            
        if isinstance(box_preds_list, list):
            box_preds = torch.cat(box_preds_list, dim=1)
        else:
            box_preds = box_preds_list
        
        # Reshape
        batch_size = x.size(0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_classes)  # [B, N, C]
        box_preds = box_preds.view(batch_size, -1, 4)  # [B, N, 4]
        
        return cls_preds, box_preds
