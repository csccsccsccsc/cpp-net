import os
import argparse

import numpy as np
from glob import glob
from skimage.io import imread
from csbdeep.utils import normalize
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap, ray_angles

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cpp_net import CPPNet

import math
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from stats_utils import get_fast_aji, get_fast_pq, get_fast_dice_2, get_dice_1, remap_label
from metric_v2 import matching, matching_dataset

try:
    import numpy_gpu as npgpu
except:
    print('The package "numpy_gpu" is used for comparison only. You can try to use the package "numpy_gpu", but it is not necessary.')


ap_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def predict_each_image(
    model_dist, img, 
    axis_norm=[0,1], center_prob_thres=0.4, seg_prob_thres=0.5, 
    n_rays=32, FPP=True, sin_angles=None, cos_angles=None, dist_cmp='cuda'
):
    
    division = 1

    img = img.copy()
    img = normalize(img, 1, 99.8, axis = axis_norm)

    h, w = img.shape
    if h%division!=0 or w%division!=0:
        dh = (h//division+1) * division - h
        dw = (w//division+1) * division - w
        img = np.pad(img, ((0, dh), (0, dw)), 'constant')

    assert(dist_cmp in ['cuda', 'cpu', 'np', 'npcuda'])

    input = torch.tensor(img)
    input = input.unsqueeze(0).unsqueeze(0)
    preds = model_dist(input.cuda())
    dist_cuda = preds[0][-1][:, :, :h, :w]
    dist = dist_cuda.data.cpu()
    prob = preds[1][-1].data.cpu()[:, :, :h, :w]
    seg = preds[2][-1].data.cpu()[:, :, :h, :w]

    dist_numpy = dist.numpy().squeeze()
    prob_numpy = prob.numpy().squeeze()
    seg = seg.numpy().squeeze()
    prob_numpy = prob_numpy*seg# (seg>=seg_prob_thres).astype(np.float32)
    
    dist_numpy = np.transpose(dist_numpy,(1,2,0))
    coord = dist_to_coord(dist_numpy)
    points = non_maximum_suppression(coord, prob_numpy, prob_thresh=center_prob_thres)
    star_label = polygons_to_label(coord, prob_numpy, points)

    # st0 = time.time()
    # You can try different approaches to finish the process of distance calculation. In our experiments dist_cmp='cuda' seems faster
    if FPP and sin_angles is None:
        if dist_cmp == 'cuda':
            angles = torch.arange(n_rays).float()/float(n_rays)*math.pi*2.0 # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1)
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1)
            sin_angles = sin_angles.cuda()
            cos_angles = cos_angles.cuda()

            offset_ih = sin_angles * dist_cuda
            offset_iw = cos_angles * dist_cuda
            # 1, r, h, w, 2
            offsets = torch.stack([offset_iw, offset_ih], dim=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(dim=1).data.cpu().squeeze(dim=0).numpy()).astype(np.int16)
        elif dist_cmp == 'cpu':
            angles = torch.arange(n_rays).float()/float(n_rays)*math.pi*2.0 # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1)
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1)

            offset_ih = sin_angles * dist
            offset_iw = cos_angles * dist
            # 1, r, h, w, 2
            offsets = torch.stack([offset_iw, offset_ih], dim=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(dim=1).data.cpu().squeeze(dim=0).numpy()).astype(np.int16)
        elif dist_cmp == 'np':
            angles = torch.arange(n_rays).float()/float(n_rays)*math.pi*2.0 # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1).data.numpy()
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1).data.numpy()

            offset_ih = sin_angles * dist.numpy()
            offset_iw = cos_angles * dist.numpy()
            # 1, r, h, w, 2
            offsets = np.stack([offset_iw, offset_ih], axis=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(axis=1).squeeze(axis=0)).astype(np.int16)
        elif dist_cmp == 'npcuda':
            angles = torch.arange(n_rays).float()/float(n_rays)*math.pi*2.0 # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1).data.numpy()
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1).data.numpy()

            offset_ih = npgpu.dot(sin_angles, dist.numpy())
            offset_iw = npgpu.dot(cos_angles, dist.numpy())
            # 1, r, h, w, 2
            offsets = np.stack([offset_iw, offset_ih], axis=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(axis=1).squeeze(axis=0)).astype(np.int16)

    pred = star_label

    # Offset-based Post Processing:
    if FPP:
        seg_remained = np.logical_and(seg>=seg_prob_thres, pred==0)
        while seg_remained.any():
            if seg_remained.any():
                rxs, rys = np.where(seg_remained)
                mean_coord_remained = mean_coord[seg_remained, :]
                pred_0 = pred.copy()
                rxs_a = np.clip((rxs + mean_coord_remained[:, 1]).astype(np.int16), 0, h-1)
                rys_a = np.clip((rys + mean_coord_remained[:, 0]).astype(np.int16), 0, w-1)
                pred[seg_remained] = pred[(rxs_a, rys_a)]
                if not((pred_0 != pred).any()):
                    break
            else:
                break
            seg_remained = np.logical_and(seg>=seg_prob_thres, pred==0)


    return pred

def run(
    DATASET_PATH_IMAGE, DATASET_PATH_LABEL,
    nc_in, n_rays, n_sampling, model_weight_path,
    center_prob_thres=0.4, seg_prob_thres=0.5
):

    X = sorted(glob(DATASET_PATH_IMAGE))
    X = list(map(imread,X))
    Y = sorted(glob(DATASET_PATH_LABEL))
    Y = list(map(imread,Y))
    
    with torch.no_grad():

        erosion_factor_list = [float(i+1)/n_sampling for i in range(n_sampling)]
        model_dist = CPPNet(nc_in, n_rays, erosion_factor_list=erosion_factor_list).cuda()
        model_dist.load_state_dict(torch.load(model_weight_path))
        model_dist.eval()

        ajis = []
        pqs = []
        dice2s = []
        dice1s = []
        aps_perimg = [[] for i_t in range(len(ap_ious))]
        preds = []

        
        for idx, img_target in enumerate(zip(X,Y)):

            image, target = img_target
            h, w = image.shape

            star_label = predict_each_image(
                model_dist, image, (0, 1),
                center_prob_thres=center_prob_thres, seg_prob_thres=seg_prob_thres, n_rays=n_rays
            )

            aji = get_fast_aji(target, star_label)
            pq = get_fast_pq(target, star_label)[0][2]
            dice2 = get_fast_dice_2(target, star_label)
            dice1 = get_dice_1(target, star_label)

            idx_aps = []
            for i_t, t in enumerate(ap_ious):
                i_t_ap = matching(target, star_label, thresh=t).accuracy
                aps_perimg[i_t].append(i_t_ap)
                idx_aps.append(i_t_ap)

            ajis.append(aji)
            pqs.append(pq)
            dice1s.append(dice1)
            dice2s.append(dice2)
            preds.append(star_label)

            print('{:03d}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(idx, aji, pq, np.mean(idx_aps), dice2, dice1))

        stats = [matching_dataset(Y, preds, thresh=t, show_progress=False, by_image=True) for t in tqdm(ap_ious)]

        avg = 0.0
        aps_perimg = np.array(aps_perimg)
        for iou in ap_ious:
            print(iou, stats[ap_ious.index(iou)].accuracy)
            avg += stats[ap_ious.index(iou)].accuracy
        avg /= len(ap_ious)
        print('avg : {:.6f}'.format(avg))
        print('aji: {:.6f}'.format(np.mean(ajis)))
        print('pq: {:.6f}'.format(np.mean(pqs)))
        print('dice2: {:.6f}'.format(np.mean(dice2s)))
        print('dice1: {:.6f}'.format(np.mean(dice1s)))

        # np.save('predictions.npy', preds)



DATASET_PATH_IMAGE = 'DATA PATH/test/images/*.tif'
DATASET_PATH_LABEL = 'DATA PATH/test/masks/*.tif'

MODEL_WEIGHT_PATH = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--nc_in', type=int, default=1)
    parser.add_argument('--n_rays', type=int, default=32)
    parser.add_argument('--n_sampling', type=int, default=6)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    run(
        DATASET_PATH_IMAGE, DATASET_PATH_LABEL,
        args.nc_in, args.n_rays, args.n_sampling, MODEL_WEIGHT_PATH,
    )
