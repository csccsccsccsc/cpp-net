# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts_gn import *
import torch.nn.init as init

class UNetStar(nn.Module):
    def __init__(self, n_channels, n_features=32, loss_type='others'):
        super(UNetStar, self).__init__()
        self.inc = inconv(n_channels, n_features)
        self.down1 = down(n_features, n_features*2)
        self.down2 = down(n_features*2, n_features*4)
        self.down3 = down(n_features*4, n_features*8)

        self.up1 = up_single(n_features*8, n_features*4, bilinear=True)
        self.up2 = up_single(n_features*4, n_features*2, bilinear=True)
        self.up3 = up_single(n_features*2, n_features*1, bilinear=True)

        self.loss_type = loss_type
        if self.loss_type=='others' or self.loss_type=='segbnd':
            self.features_segbnd = nn.Conv2d(n_features, n_features, 3, padding=1)
            self.out_segbnd = outconv(n_features, 2)
        if self.loss_type=='others' or self.loss_type=='bbox':
            self.features_bbox = nn.Conv2d(n_features, n_features, 3, padding=1)
            self.out_bbox = outconv(n_features, 4)

        self.final_activation_prob = nn.Sigmoid()
        self.final_activation_ray = nn.ReLU()

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        
        if self.loss_type == 'others' or self.loss_type=='segbnd':
            x_segbnd = self.final_activation_prob(self.out_segbnd(self.features_segbnd(x)))
        if self.loss_type == 'others' or self.loss_type=='bbox':
            x_bbox = self.final_activation_ray(self.out_bbox(self.features_bbox(x)))

        if self.loss_type == 'others':
            return x_segbnd, x_bbox
        elif self.loss_type == 'segbnd':
            return x_segbnd, 0.0
        elif self.loss_type == 'bbox':
            return 0.0, x_bbox