import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.transform import resize
import numpy as np
from stardist import star_dist,edt_prob
from csbdeep.utils import normalize
import random
import scipy.io as scio

class PanNukeDataset(Dataset):
    def __init__(self, root_dir, n_rays, max_dist=None, if_training=False, resz=None):
        self.img_filefold = os.path.join(root_dir,'images')
        self.target_filefold =  os.path.join(root_dir,'masks')

        self.img_list = []
        self.target_list = []
        with open(os.path.join(self.img_filefold, 'name_list.txt'), 'r') as f:
            for line in f.readlines():
                line_terms = line.split(',')
                self.img_list.append(line_terms[1].strip())
        for ic in range(0, 5):
             with open(os.path.join(self.target_filefold, 'name_list_c'+str(ic)+'.txt'), 'r') as f:
                ic_target_list = []
                for line in f.readlines():
                    line_terms = line.split(',')
                    ic_target_list.append(line_terms[1].strip())
                self.target_list.append(ic_target_list)

        self.n_rays = n_rays
        self.max_dist = max_dist
        self.if_training=if_training
        self.resz = resz

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_list[idx])
        image = image.astype(np.float32)
        for imod in range(3):
            tmp_image = image[:, :, imod]
            meanv = tmp_image.mean()
            stdv = tmp_image.std()
            image[:, :, imod] = (tmp_image-meanv)/stdv
        # target: [0, 4]
        target = []
        for ic in range(5):
            ic_target = io.imread(self.target_list[ic][idx])
            if ic > 0:
                last_target_max = target[ic-1].max()
                ic_target[ic_target>0] += last_target_max
            target.append(ic_target)
        target = np.stack(target, axis=2)
        if self.if_training:
            aug_type = random.randint(0, 5) # rot90: 0, 1, 2; flip: 3, 4; ori: 5
            if aug_type<=2:
                image = np.rot90(image, k=aug_type+1, axes=(0, 1)).copy()
                target = np.rot90(target, k=aug_type+1, axes=(0, 1)).copy()
            elif aug_type<=4:
                image = np.flip(image, axis=aug_type-3).copy()
                target = np.flip(target, axis=aug_type-3).copy()
        obj_probabilities = []
        allcls_target = target.max(axis=2)
        distances = star_dist(allcls_target, self.n_rays)
        if self.max_dist:
            distances[distances>self.max_dist] = self.max_dist
        obj_probabilities = edt_prob(allcls_target)
        # seg + 1 !!!
        # background = 0; foreground = 1:5
        seg_target = ((np.argmax(target, axis=2)+1).astype(np.int64) * (target.max(axis=2)>0).astype(np.int64)).astype(np.int64)

        if self.resz is not None:
            image = resize(image, self.resz, order=1, preserve_range=True)
            obj_probabilities = resize(obj_probabilities, self.resz, order=1, preserve_range=True)
            distances = resize(distances, self.resz, order=1, preserve_range=True)
            seg_target = resize(seg_target, self.resz, order=0, preserve_range=True)
        image = np.transpose(image, (2, 0, 1))
        distances = np.transpose(distances, (2,0,1))
        obj_probabilities = np.expand_dims(obj_probabilities, axis=0)
        # seg_target = np.expand_dims(seg_target, axis=0)
        # scio.savemat('test.mat', {'image':image, 'prob':obj_probabilities, 'dist':distances, 'target':target, 'seg':seg_target})
        # print(self.img_list[idx])
        return image, obj_probabilities, distances, seg_target

def getDataLoaders(n_rays, max_dist, root_dir, type_list=['fold_1', 'fold_2'], batch_size=8, resz=None):
    trainset = PanNukeDataset(root_dir=root_dir+'/'+type_list[0]+'/', n_rays=n_rays, max_dist=max_dist, if_training=True, resz=resz)
    testset = PanNukeDataset(root_dir=root_dir+'/'+type_list[1]+'/', n_rays=n_rays, max_dist=max_dist, if_training=False, resz=resz)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader,testloader