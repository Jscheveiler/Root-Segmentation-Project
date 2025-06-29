# Use dilation = 1,2,4,6, or 1,3,5,7
# It's in the bottleneck, so the dimensions are H/16, W/16, C=1024.
# The base values were going up to 18, but 3+(2*17) = 37x37 which is way too big, it's more than 100% of the feature map
# So we reduce the dilations.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """
    Implementation of Atrous Spatial Pyramid Pooling.
    see : "Rethinking Atrous Convolution for Semantic Image Segmentation", L. Chen et al., 2017:
    https://arxiv.org/pdf/1706.05587

    The dilations used are : 1, 2, 4, 6 instead of the default 1, 6, 12, 18.
    This choice is motivated by the position of ASPP in the U-Net model. It is placed at the bottleneck,
    and receives features maps of size (B, 1024, 32, 32). Therefore, high dilation values aren't suitable
    because they will mostly compute padding.
    For example, a 3x3 kernel with dilation=18 has a receptive field of 37x37, which covers 116% of the feature map.
    Most sampling points would end up in the padding.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1x1 Conv2D is sufficient for the first block since its focus is set on local features.
        # bias is set to False because GroupNorm is added right after the Conv layer.
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_block4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block2(x)
        x3 = self.atrous_block4(x)
        x4 = self.atrous_block6(x)

        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=False)

        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.fuse(x_cat)