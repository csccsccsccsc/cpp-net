from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os
import matplotlib
from tqdm import tqdm
matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob
#from tifffile import imread
from skimage.io import imread
from skimage.measure import label
from csbdeep.utils import normalize
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap, ray_angles
import torch
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as scio
from scipy import stats
import math
import time
# from models.cpp_net import CPPNet
from models.cpp_net_res50 import CPPNet

from stats_utils import remap_label

import warnings
warnings.filterwarnings("ignore")

def predict_each_image(model_dist, img, resz=None, offset_refine=True, TARGET_LABELS=32, prob_thres=0.4):

    img = img.astype(np.float32)
    for imod in range(3):
        tmp_img = img[:, :, imod]
        meanv = tmp_img.mean()
        stdv = tmp_img.std()
        img[:, :, imod] = (tmp_img-meanv)/stdv

    input = torch.tensor(img)
    input = input.unsqueeze(0)
    if len(input.shape) < 4:
        input = input.unsqueeze(1)
    else:
        input = input.permute([0, 3, 1, 2])
    _, _, h, w = input.shape

    # Model Prediction
    if resz is not None:
        resz_input = F.interpolate(input.cuda(), size=resz, mode='bilinear', align_corners=True)
    else:
        resz_input = input.cuda()
    preds = model_dist(resz_input)
    dist = preds[0]
    prob = preds[1]
    seg = preds[2]
    if isinstance(dist, (tuple, list)):
        dist_cuda = dist[-1].clone()
        dist = dist[-1].detach().cpu()
    else:
        dist_cuda = dist.clone()
        dist = dist.detach().cpu()
    if isinstance(prob, (tuple, list)):
        prob = prob[-1].detach().cpu()
    else:
        prob = prob.detach().cpu()
    if isinstance(seg, (tuple, list)):
        seg = seg[-1].detach().cpu()
    else:
        seg = seg.detach().cpu()
    seg = F.softmax(seg, dim=1)
    if resz is not None:
        dist_cuda = F.interpolate(dist_cuda, size=[h, w], mode='bilinear', align_corners=True)
        dist = F.interpolate(dist, size=[h, w], mode='bilinear', align_corners=True)
        prob = F.interpolate(prob, size=[h, w], mode='bilinear', align_corners=True)
        seg = F.interpolate(seg, size=[h, w], mode='bilinear', align_corners=True)
    dists = dist.numpy().squeeze()
    probs = prob.numpy().squeeze()
    segs = seg.numpy().squeeze()

    # Post Processing
    dists = np.transpose(dists,(1,2,0))
    coord = dist_to_coord(dists)
    points = non_maximum_suppression(coord, probs, prob_thresh=prob_thres)
    binary_star_label = polygons_to_label(coord, probs, points)
    binary_star_label = remap_label(binary_star_label)

    # segs: background + n_cls foregrund
    # cls_star_labels: n_cls foreground
    N_CLASSES = segs.shape[0]
    seg_label = np.argmax(segs, axis=0)
    cls_star_labels = np.zeros((N_CLASSES-1, )+binary_star_label.shape, dtype=np.int16)
    cset = np.unique(binary_star_label[binary_star_label>0])
    for ic in cset:
        icmap = binary_star_label==ic
        ic_seg_label = seg_label[icmap]
        ic_cls = stats.mode(ic_seg_label)[0][0]
        if ic_cls>0:
            cls_star_labels[ic_cls-1][icmap] = ic

    if offset_refine:
        h, w, n_rays = dists.shape
        angles = torch.arange(n_rays).float()/float(n_rays)*math.pi*2.0 # 0 - 2*pi
        sin_angles = torch.sin(angles).view(1, n_rays, 1, 1).to(dist_cuda.device)
        cos_angles = torch.cos(angles).view(1, n_rays, 1, 1).to(dist_cuda.device)
        offset_ih = sin_angles * dist_cuda
        offset_iw = cos_angles * dist_cuda
        # 1, r, h, w, 2
        offsets = torch.stack([offset_iw, offset_ih], dim=-1)
        # h, w, 2
        mean_coord = np.round(offsets.mean(dim=1).data.cpu().squeeze(dim=0).numpy()).astype(np.int16)

    for icls in range(N_CLASSES-1):
        icls_star_label = cls_star_labels[icls]
        binary_seg = seg_label==(icls+1)
        if icls_star_label.any():
            if offset_refine:
                seg_remained = np.logical_and(binary_seg, icls_star_label==0)
                pred = icls_star_label.copy()
                i_iter = 0
                while seg_remained.any():
                    i_iter += 1
                    rxs, rys = np.where(seg_remained)
                    pred_0 = pred.copy()
                    for rx, ry in zip(rxs, rys):
                        dx_rx, dy_rx = np.clip(int(np.round(rx + mean_coord[rx, ry, 1])), 0, h-1), np.clip(int(np.round(ry + mean_coord[rx, ry, 0])), 0, w-1)
                        pred[rx, ry] = pred[dx_rx, dy_rx]
                    if not((pred_0 != pred).any()):
                        break
                    seg_remained = np.logical_and(binary_seg, pred==0)
                icls_star_label = pred
        if len(np.unique(icls_star_label[icls_star_label>0])) >= 1:
            icls_star_label = remap_label(icls_star_label)
        cls_star_labels[icls] = icls_star_label
    cls_star_labels = cls_star_labels.transpose([1, 2, 0])

    return cls_star_labels

# Classification:
# GT: 0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 6: Background
# Pred: 0: Background, 1: Neoplastic cells, 2: Inflammatory, 3: Connective/Soft tissue cells, 4: Dead Cells, 5: Epithelial


nk = 6
erosion_factor_list = [float(i+1)/nk for i in range(nk)]

offset_refine = True

if not offset_refine:
    results_filefold = './PanNuke_aug_results_x1_0806/'
    if not os.path.exists(results_filefold):
        os.makedirs(results_filefold)
else:
    results_filefold = './PanNuke_aug_results_offset_refine_x1_0806/'
    if not os.path.exists(results_filefold):
        os.makedirs(results_filefold)

def run(
    DATASET_PATH_IMAGE, model_weight_path, prediction_save_path,
    nc_in=3, n_rays=32, n_sampling=6, n_cls=6,
    center_prob_thres=0.4, resz=None,
):
    image_name_list = []
    with open(os.path.join(DATASET_PATH_IMAGE, 'name_list.txt'), 'r') as f_img:
        for line in f_img.readlines():
            line_term = (line.split(',')[-1]).strip()
            image_name_list.append(line_term)

    n_data = len(image_name_list)
    preds = np.zeros([n_data, 256, 256, N_CLASSES], dtype=np.int16)
    with torch.no_grad():
        erosion_factor_list = [float(i+1)/n_sampling for i in range(n_sampling)]
        model_dist = CPPNet(nc_in, n_rays, erosion_factor_list=erosion_factor_list, n_seg_cls=n_cls)
        model_dist = model_dist.cuda()
        model_dist.load_state_dict(torch.load(model_weight_path))
        model_dist.eval()
        for idx, image_name in enumerate(image_name_list):
            image = imread(image_name)
            pred = predict_each_image(model_dist, image, resz=resz, offset_refine=offset_refine, prob_thres=center_prob_thres)
            preds[idx] = pred

    np.save(prediction_save_path, preds)


DATASET_PATH_IMAGE = 'DATA PATH/test/images/*.tif or .png'
PREDICTION_SAVE_PATH = 'SAVE_PATH/*.npy'
MODEL_WEIGHT_PATH = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--nc_in', type=int, default=3)
    parser.add_argument('--n_rays', type=int, default=32)
    parser.add_argument('--n_sampling', type=int, default=6)
    parser.add_argument('--n_cls', type=int, default=6)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    run(
        DATASET_PATH_IMAGE, MODEL_WEIGHT_PATH, PREDICTION_SAVE_PATH,
        args.nc_in, args.n_rays, args.n_sampling, args.n_cls, 
    )
