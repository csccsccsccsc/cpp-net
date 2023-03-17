import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts_gn import *
from .SamplingFeatures2 import SamplingFeatures

class CPPNet(nn.Module):

    def __init__(self, n_channels, n_rays, erosion_factor_list=[0.2, 0.4, 0.6, 0.8, 1.0], return_conf=False, with_seg=True):
        super(CPPNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        self.up1 = up(256, 64, bilinear=True)
        self.up2 = up(128, 32, bilinear=True)
        self.up3 = up(64, 32, bilinear=True)
        self.features = nn.Conv2d(32, 128, 3, padding=1)
        self.out_prob = outconv(128, 1)
        self.out_ray = outconv(128, n_rays)
        self.conv_0_confidence = outconv(128, n_rays)
        self.conv_1_confidence = outconv(1+len(erosion_factor_list), 1+len(erosion_factor_list))
        nn.init.constant_(self.conv_1_confidence.conv.bias, 1.0)

        self.with_seg = with_seg
        if self.with_seg:
            self.up1_seg = up(256, 64, bilinear=True)
            self.up2_seg = up(128, 32, bilinear=True)
            self.up3_seg = up(64, 32, bilinear=True)
            self.out_seg = outconv(32, 1)

        self.final_activation_ray = nn.ReLU()
        self.final_activation_prob = nn.Sigmoid()

        # Refinement
        self.sampling_feature = SamplingFeatures(n_rays)
        self.erosion_factor_list = erosion_factor_list
        self.n_rays = n_rays
        self.return_conf = return_conf

    def forward(self, img, gt_dist=None):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.features(x)
        out_ray = self.out_ray(x)
        out_confidence = self.conv_0_confidence(x)
        out_prob = self.out_prob(x)

        if gt_dist is not None:
            out_ray_for_sampling = gt_dist
        else:
            out_ray_for_sampling = out_ray
        ray_refined = [ out_ray_for_sampling ]

        confidence_refined = [ out_confidence ]
        for erosion_factor in self.erosion_factor_list:
            base_dist = (out_ray_for_sampling-1.0)*erosion_factor
            ray_sampled, _ = self.sampling_feature(out_ray_for_sampling, base_dist, 1)
            conf_sampled, _ = self.sampling_feature(out_confidence, base_dist, 1)
            ray_refined.append(ray_sampled + base_dist)
            confidence_refined.append(conf_sampled)
        ray_refined = torch.stack(ray_refined, dim=1)
        b, k, c, h, w = ray_refined.shape

        confidence_refined = torch.stack(confidence_refined, dim=1)
        #confidence_refined = torch.cat((confidence_refined, ray_refined), dim=1)
        confidence_refined = confidence_refined.permute([0,2,1,3,4]).contiguous().view(b*c, k, h, w)
        confidence_refined = self.conv_1_confidence(confidence_refined)
        confidence_refined = confidence_refined.view(b, c, k, h, w).permute([0,2,1,3,4])
        confidence_refined = F.softmax(confidence_refined, dim=1)
        if self.return_conf:
            out_conf = [out_confidence, confidence_refined]
        else:
            out_conf = None
        ray_refined = (ray_refined*confidence_refined).sum(dim=1)

        out_ray = self.final_activation_ray(out_ray)
        ray_refined = self.final_activation_ray(ray_refined)
        out_prob = self.final_activation_prob(out_prob)


        if self.with_seg:
            x_seg = self.up1_seg(x4, x3)
            x_seg = self.up2_seg(x_seg, x2)
            x_seg = self.up3_seg(x_seg, x1)
            out_seg = self.out_seg(x_seg)
            out_seg = self.final_activation_prob(out_seg)
        else:
            out_seg = None


        return [out_ray, ray_refined], [out_prob], [out_seg, ], [out_conf, ]


    def init_weight(self):
        for m in self.modules():        
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.conv_1_confidence.conv.bias, 1.0)
