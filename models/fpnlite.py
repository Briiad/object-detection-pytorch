import torch
from torch import nn

class FPNLite(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPNLite, self).__init__()
        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
    
    def forward(self, x):
        # Apply FPN to each layer output from the backbone
        return [fpn_layer(feat) for fpn_layer, feat in zip(self.fpn_layers, x)]
