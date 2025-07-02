import torch
import torch.nn as nn
from models.layers.se_block import SEBlock

class ResidualDoubleConv(nn.Module):
    """
    Residual Block for a U-Net. It is made of :
        -> 3x3 Conv2D layer with a padding of 1px (To avoid the dimensional shrinking of the convolution)
        This layer halves the channel count
        -> Group Normalization with 8 groups by default (so that there is always at least 8 channels per group (given
        that our lowest channel count is 64, and 64/8 = 8) as the results from Wu & He, 2018, seem to indicate that
         the performance drops under 8 channels per group. (https://arxiv.org/pdf/1803.08494, Table 3, Bottom row)
        -> ReLU activation
        -> 3x3 Conv2D layer with a padding of 1px. This layer doesn't change channel count.
        -> Group Normalization with 8 groups.

        -> SE Block on the output of the convolution block so that the feature maps get weighed, allowing
        the model to learn what feature is more meaningful than others.
        -> ReLU activation
        -> The whole block is a residual block.
    """
    def __init__(self, in_channels, out_channels, groups=8, up=False):
        super().__init__()

        self.conv_block = nn .Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels)
        )

        if not up:
            self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual needs C_in and C_out to be equal, because it won't be possible to merge the Identity with the Output
        # if their dimensions differ. Therefore, a 1x1 convolution is implemented to make their dimensions match.
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)  # Define our identity and make it match output dim if needed

        out = self.conv_block(x)
        if not up:
            out = self.se(out)

        return self.relu(out + residual)
