import torch
from torch import nn

# In ssd.py

class SSDHead(nn.Module):
    def __init__(self, num_classes, feature_sizes=[80, 40, 20]):
        super().__init__()
        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        
        # Separate convs for each feature level
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
            for _ in feature_sizes
        ])
        
        self.box_heads = nn.ModuleList([
            nn.Conv2d(256, 4, kernel_size=3, padding=1) 
            for _ in feature_sizes
        ])

    def forward(self, features):
        cls_preds = []
        box_preds = []
        
        # Process each feature level
        for feat, cls_head, box_head in zip(features, self.cls_heads, self.box_heads):
            # Get predictions
            cls_pred = cls_head(feat)
            box_pred = box_head(feat)
            
            # Reshape to (batch, -1, num_classes/4)
            batch = cls_pred.size(0)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(batch, -1, self.num_classes)
            
            box_pred = box_pred.permute(0, 2, 3, 1).contiguous()
            box_pred = box_pred.view(batch, -1, 4)
            
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
        
        # Concatenate along anchor dimension
        cls_preds = torch.cat(cls_preds, dim=1)
        box_preds = torch.cat(box_preds, dim=1)
        
        return cls_preds, box_preds