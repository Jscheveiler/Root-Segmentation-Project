"""
This file contains a modified U-Net network which implements Residual blocks with SE blocks, and an ASPP block as the bottleneck.
It also uses group normalization instead of Batch Normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.residual_block import ResidualDoubleConv
from models.layers.se_block import SEBlock
from models.layers.aspp import ASPP

from models.layers.double_conv import Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Define the Encoder
        self.inc = ResidualDoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Add ASPP Bottleneck
        self.aspp = ASPP(1024, 1024)

        # Define the Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, out_channels)

    def forward(self, img):
        # Pass in the encoder
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Pass x5 in the ASPP bottleneck
        x5 = self.aspp(x5)

        # Pass in the decoder
        # Each layer of the decoder takes as input the output of the previous layer + the output of the corresponding
        # encoder layer. (Skips connection of the U-Net architecture)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)  # No sigmoid call here, it's handled in the loss function.
