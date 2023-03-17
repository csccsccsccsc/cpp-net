# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts_gn import *
import torch.nn.init as init

# The encoder only
class Feature_Extractor(nn.Module):
    def __init__(self, n_channels, n_features=32):
        super(Feature_Extractor, self).__init__()
        self.inc = inconv(n_channels, n_features)
        self.down1 = down(n_features, n_features*2)
        self.down2 = down(n_features*2, n_features*4)
        self.down3 = down(n_features*4, n_features*8)
        self.down4 = down(n_features*8, n_features*16)

        # self.up1 = up_single(n_features*16, n_features*8, bilinear=True)
        # self.up2 = up_single(n_features*8, n_features*4, bilinear=True)
        # self.up3 = up_single(n_features*4, n_features*2, bilinear=True)
        # self.up4 = up_single(n_features*2, n_features*1, bilinear=True)

        # self.features_segbnd = nn.Conv2d(n_features, n_features, 3, padding=1)
        # self.features_bbox = nn.Conv2d(n_features, n_features, 3, padding=1)
        # self.out_segbnd = outconv(n_features, 2)
        # self.out_bbox = outconv(n_features, 4)
        # self.final_activation_prob = nn.Sigmoid()
        # self.final_activation_ray = nn.ReLU()

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # x = self.up1(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # x = self.up4(x, x0)

        # x_segbnd = self.final_activation_prob(self.out_segbnd(self.features_segbnd(x)))
        # x_bbox = self.final_activation_ray(self.out_bbox(self.features_bbox(x)))

        return x4