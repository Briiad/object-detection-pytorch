import torch
from torch import nn

class FPNLite(nn.Module):
    def __init__(self, input_channels_list, output_channels=256):
        super(FPNLite, self).__init__()
        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(in_channels, output_channels, kernel_size=1)
            for in_channels in input_channels_list
        ])

    def forward(self, x):
        return [fpn_layer(feat) for fpn_layer, feat in zip(self.fpn_layers, x)]


