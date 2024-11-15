import torch
from torch import nn

class SSDHead(nn.Module):
    def __init__(self, num_classes, num_boxes=6):
        super(SSDHead, self).__init__()
        self.cls_head = nn.Conv2d(256, num_classes * num_boxes, kernel_size=3, padding=1)
        self.box_head = nn.Conv2d(256, 4 * num_boxes, kernel_size=3, padding=1)

    def forward(self, features):
        cls_preds, box_preds = [], []
        for feature in features:
            cls_preds.append(self.cls_head(feature))
            box_preds.append(self.box_head(feature))
        return cls_preds, box_preds
