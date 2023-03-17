import torch
import torch.nn.functional as F
import torch.nn as nn
import math
#import time

def feature_sampling(feature_map, coord_map, nd_sampling, sampling_mode='nearest'):
    b, c, h, w = feature_map.shape
    # coord_map: b, k, 2, h, w
    # 'k' for k rays in each image
    _, k, _, h, w = coord_map.shape
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    # grid: b, 1, 2, h, w
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1).cuda()
    # sampling_coord: b, k, 2, h, w
    sampling_coord = grid + coord_map
    sampling_coord[:, :, 0, :, :] = sampling_coord[:, :, 0, :, :]/(w-1)
    sampling_coord[:, :, 1, :, :] = sampling_coord[:, :, 1, :, :]/(h-1)
    sampling_coord = sampling_coord*2.0-1.0

    assert(k*nd_sampling==c)

    if nd_sampling > 0:
        sampling_coord = sampling_coord.permute(1, 0, 3, 4, 2).flatten(start_dim=0, end_dim=1) # kb, h, w, 2
        sampling_features = F.grid_sample(feature_map.view(b, k, nd_sampling, h, w).permute(1, 0, 2, 3, 4).flatten(start_dim=0, end_dim=1), sampling_coord, mode=sampling_mode) # kb, c', h, w
        sampling_features = sampling_features.view(k, b, nd_sampling, h, w).permute(1, 0, 2, 3, 4) # b, k, c', h, w
    else:
        sampling_coord = sampling_coord.permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=2) # b, kh, w, 2
        sampling_features = F.grid_sample(feature_map, sampling_coord, mode=sampling_mode)
        sampling_features = sampling_features.view(b, c, k, h, w).permute(0, 2, 1, 3, 4) # b, k, c'/c, h, w

    sampling_features = sampling_features.flatten(start_dim=1, end_dim=2) # b, k*c', h, w

    return sampling_features, sampling_coord

class SamplingFeatures(nn.Module):
    def __init__(self, n_rays, sampling_mode='nearest'):
        super(SamplingFeatures, self).__init__()
        self.n_rays = n_rays
        self.angles = torch.arange(n_rays).float()/float(n_rays)*math.pi*2.0 # 0 - 2*pi
        self.sin_angles = torch.sin(self.angles).cuda().view(1, n_rays, 1, 1)
        self.cos_angles = torch.cos(self.angles).cuda().view(1, n_rays, 1, 1)
        self.sampling_mode = sampling_mode
    def forward(self, feature_map, dist, nd_sampling):
        # feature_map: b, c, h, w
        # dist: b, k, h, w
        # sampled_features: b, k*c, h, w
        offset_ih = self.sin_angles * dist
        offset_iw = self.cos_angles * dist
        offsets = torch.stack([offset_iw, offset_ih], dim=2)
        sampled_features, sampling_coord = feature_sampling(feature_map, offsets, nd_sampling, self.sampling_mode)
        return sampled_features, sampling_coord
