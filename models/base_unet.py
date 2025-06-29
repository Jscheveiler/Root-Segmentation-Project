"""
This file contains a "basic" version of a U-Net network, used to compare the modified version to a more classical approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDoubleConv(nn.Module):
    """
    Basic DoubleConv Module
    Conv2d => BatchNorm => ReLU => Conv2D => BatchNorm => ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DownBlock(nn.Module):
    """
    Basic DownBlock Module
    2x2 MaxPool2D => BaseDoubleConv

    MaxPool2D uses a 2x2 kernel to halve the resolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            BaseDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """
    Basic UpBlock Module
    2x2 Upsampling => BaseDoubleConv

    Upsampling halves the number of channels and increases resolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The upsampling reduces channels by half
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation: (in_channels // 2) + out_channels -> out_channels
        self.conv = BaseDoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Ensure spatial size match due to rounding in pooling
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    Basic UNet Network
    4 Downblocks, 1 Bottleneck, 4 Upblocks.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Define the Encoder
        self.inc = BaseDoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        # Add Bottleneck
        self.bottleneck = BaseDoubleConv(1024, 1024)

        # Define the Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.outc = BaseDoubleConv(64, out_channels)

    def forward(self, img):
        # Pass in the encoder
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Pass x5 in the bottleneck
        x5 = self.bottleneck(x5)

        # Pass in the decoder
        # Each layer of the decoder takes as input the output of the previous layer + the output of the corresponding
        # encoder layer. (Skips connection of the U-Net architecture)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # No sigmoid call here, it's handled in the loss function.