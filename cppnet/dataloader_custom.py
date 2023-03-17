import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.transform import resize
import numpy as np
from stardist import star_dist,edt_prob
from csbdeep.utils import normalize
import random

class DSB2018Dataset(Dataset):
    def __init__(self, root_dir, n_rays, max_dist=None, if_training=False, resz=None, crop=None, image_flag='images', mask_flag='masks'):
        self.raw_files = os.listdir(os.path.join(root_dir, image_flag))
        self.target_files = os.listdir( os.path.join(root_dir, mask_flag))
        self.raw_files.sort()
        self.target_files.sort()
        self.root_dir = root_dir
        self.n_rays = n_rays
        self.max_dist = max_dist
        self.if_training=if_training
        self.resz = resz
        self.crop = crop

        self.image_flag = image_flag
        self.mask_flag = mask_flag

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, idx):
        assert self.raw_files[idx] == self.target_files[idx]
        img_name = os.path.join(self.root_dir, self.image_flag, self.raw_files[idx])
        image = io.imread(img_name)
        target_name = os.path.join(self.root_dir, self.mask_flag, self.target_files[idx])
        target = io.imread(target_name)

        if self.crop is not None and (None not in self.crop):
            h, w = image.shape
            dh = h-self.crop[0]
            dw = w-self.crop[1]
            sh = random.randint(0, dh-1)
            sw = random.randint(0, dw-1)
            image = image[sh:(sh+self.crop[0]), sw:(sw+self.crop[1])]
            target = target[sh:(sh+self.crop[0]), sw:(sw+self.crop[1])]

        image = normalize(image, 1, 99.8, axis=(0,1))

        if self.if_training:
            aug_type = random.randint(0, 5) # rot90: 0, 1, 2; flip: 3, 4; ori: 5
            if aug_type<=2:
                image = np.rot90(image, aug_type).copy()
                target = np.rot90(target, aug_type).copy()
            elif aug_type<=4:
                image = np.flip(image, aug_type-3).copy()
                target = np.flip(target, aug_type-3).copy()
        distances = star_dist(target, self.n_rays)
        if self.max_dist:
            distances[distances>self.max_dist] = self.max_dist
        obj_probabilities = edt_prob(target)

        if self.resz is not None:
            image = resize(image, self.resz, order=1, preserve_range=True)
            obj_probabilities = resize(obj_probabilities, self.resz, order=1, preserve_range=True)
            distances = resize(distances, self.resz, order=1, preserve_range=True)

        distances = np.transpose(distances, (2,0,1))
        image = np.expand_dims(image,0)
        obj_probabilities = np.expand_dims(obj_probabilities,0)

        return image, obj_probabilities, distances

def getDataLoaders(n_rays, max_dist, root_dir, type_list=['train', 'val'], image_flag='images', mask_flag='masks', batch_size=1, train_crop=None, test_crop=None, resz=None):
    trainset = DSB2018Dataset(root_dir=root_dir+'/'+type_list[0]+'/', n_rays=n_rays, max_dist=max_dist, if_training=True, crop=train_crop, resz=resz, image_flag=image_flag, mask_flag=mask_flag)
    testset = DSB2018Dataset(root_dir=root_dir+'/'+type_list[1]+'/', n_rays=n_rays, max_dist=max_dist, if_training=False, crop=test_crop, resz=resz, image_flag=image_flag, mask_flag=mask_flag)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader,testloader