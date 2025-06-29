import torch
import torch.nn as nn

from models.layers.residual_block import ResidualDoubleConv

class Down(nn.Module):
    """
    Downscaling with a MaxPool2d layer, followed by a Residual DoubleConv block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """
    Upscaling then calling a Residual DoubleConv block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # Concat x1 from deeper layer and x2 from the skip connection
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution to map to desired output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
