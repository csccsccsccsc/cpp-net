import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.transform import resize
import numpy as np
from stardist import star_dist,edt_prob
from csbdeep.utils import normalize
import random
from scipy import ndimage

class DSB2018Dataset(Dataset):
    def __init__(self, root_dir, n_rays, max_dist=None, if_training=False, resz=None):
        self.raw_files = os.listdir(os.path.join(root_dir,'images'))
        self.target_files = os.listdir( os.path.join(root_dir,'masks'))
        self.raw_files.sort()
        self.target_files.sort()
        self.root_dir = root_dir
        self.n_rays = n_rays
        self.max_dist = max_dist
        self.if_training=if_training
        self.resz = resz

    def __len__(self):
        return len(self.raw_files)


    def __getitem__(self, idx):
        assert self.raw_files[idx] == self.target_files[idx]
        target_name = os.path.join(self.root_dir, 'masks', self.target_files[idx])
        target = io.imread(target_name)
        if self.if_training:
            aug_type = random.randint(0, 5) # rot90: 0, 1, 2; flip: 3, 4; ori: 5
            if aug_type<=2:
                target = np.rot90(target, aug_type).copy()
            elif aug_type<=4:
                target = np.flip(target, aug_type-3).copy()
        distances = star_dist(target, self.n_rays)
        if self.max_dist:
            distances[distances>self.max_dist] = self.max_dist
        obj_probabilities = edt_prob(target)

        if self.resz is not None:
            obj_probabilities = resize(obj_probabilities, self.resz, order=1, preserve_range=True)
            distances = resize(distances, self.resz, order=1, preserve_range=True)

        distances = np.transpose(distances, (2,0,1))
        obj_probabilities = np.expand_dims(obj_probabilities,0)
        
        seg = (target>0).astype(np.float32)
        
        h, w = target.shape
        
        cset = np.unique(target[target>0])
        bndmap = np.zeros(target.shape, dtype=np.float32)
        cxmap = np.zeros(target.shape, dtype=np.float32)
        cymap = np.zeros(target.shape, dtype=np.float32)
        chmap = np.zeros(target.shape, dtype=np.float32)
        cwmap = np.zeros(target.shape, dtype=np.float32)

        for ic in cset:
            icmap = target==ic
            bndmap += np.logical_xor(ndimage.morphology.binary_dilation(icmap, iterations=2), icmap).astype(np.float32)
            cx, cy = np.nonzero(icmap)
            cxmap[icmap] = cx.mean() / h
            cymap[icmap] = cy.mean() / w
            chmap[icmap] = cx.max()-cx.min()
            cwmap[icmap] = cy.max()-cy.min()
        bndmap[bndmap>1] = 1.0
        
        # if random.random()>=0.5:
            # sigma = random.random()*2
            # distances = ndimage.gaussian_filter(distances, sigma=sigma, mode='reflect')
            # obj_probabilities = ndimage.gaussian_filter(obj_probabilities, sigma=sigma, mode='reflect')
        input_stardist = np.concatenate((obj_probabilities, distances), axis=0)

        segbnd = np.stack((seg, bndmap), axis=0)
        bbox = np.stack((cxmap, cymap, chmap, cwmap), axis=0)

        return input_stardist, segbnd, bbox

def getDataLoaders(n_rays, max_dist, root_dir, type_list=['train', 'test'], batch_size=1, resz=None):
    trainset = DSB2018Dataset(root_dir=root_dir+'/'+type_list[0]+'/', n_rays=n_rays, max_dist=max_dist, if_training=True, resz=resz)
    testset = DSB2018Dataset(root_dir=root_dir+'/'+type_list[1]+'/', n_rays=n_rays, max_dist=max_dist, if_training=False, resz=resz)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader,testloader