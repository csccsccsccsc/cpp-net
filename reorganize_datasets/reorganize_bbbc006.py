import os
import glob
from shutil import copyfile
from skimage import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tqdm

#######
### Download PanNuke from https://data.broadinstitute.org/bbbc/BBBC006 (In CPP-Net, we only use the "BBBC006_v1_images_z_16" images.)

def image_matching(base_name, name_list):
    match_name = []
    for name in name_list:
        if name.find(base_name)!=-1:
            match_name.append(name)
    return match_name

img_dir = 'DATA ROOT PATH/bbbc006/BBBC006/BBBC006_v1_images_z_16'
msk_dir = 'DATA ROOT PATH/bbbc006/BBBC006_v1_labels'
save_img_dir = 'DATA ROOT PATH/bbbc006/BBBC006_split'
name_list = sorted(os.listdir(os.path.join(msk_dir)))
img_name_list = os.listdir(os.path.join(img_dir))

rng = np.random.RandomState(42)
ind = rng.permutation(len(name_list))
n_val = max(1, int(round(0.15 * len(ind))))
n_test = max(1, int(round(0.15 * len(ind))))
n_train = len(ind) - n_val - n_test
ind_train, ind_val, ind_test = ind[:n_train], ind[n_train:n_train+n_val], ind[n_train+n_val:]
print('number of images: %3d' % len(name_list))
print('- training:       %3d' % len(ind_train))
print('- validation:     %3d' % len(ind_val))
print('- testing:        %3d' % len(ind_test))

inds = {'train':ind_train, 'val':ind_val, 'test':ind_test}

for split_flag in ['train', 'val', 'test']:
    if not(os.path.exists(os.path.join(save_img_dir, split_flag, 'images'))):
        os.makedirs(os.path.join(save_img_dir, split_flag, 'images'))
    if not(os.path.exists(os.path.join(save_img_dir, split_flag, 'masks'))):
        os.makedirs(os.path.join(save_img_dir, split_flag, 'masks'))
    image_id = 0
    with open(os.path.join(save_img_dir, split_flag+'_name.txt'), 'w') as f:
        cur_ind = inds[split_flag]
        bar = tqdm.tqdm(range(len(cur_ind)))
        for i_name in bar:
            name = name_list[cur_ind[i_name]]
            mask_source = os.path.join(msk_dir, name)
            mask = io.imread(mask_source)
            base_name = '.'.join(name.split('.')[:-1])
            imgs_source = image_matching(base_name, img_name_list)
            count_match_img = len(imgs_source)
            for img_source in imgs_source:
                img = io.imread(os.path.join(img_dir, img_source))
                img_target = os.path.join(save_img_dir, split_flag, 'images', str(image_id)+'.tif')
                io.imsave(img_target, img)
                mask_target = os.path.join(save_img_dir, split_flag, 'masks', str(image_id)+'.tif')
                io.imsave(mask_target, mask)
                #print(image_id, img.dtype, mask.dtype, img.shape, mask.shape)
                image_id += 1

                # Filename record
                f.write(str(image_id)+','+name+','+split_flag+'\n')
                bar.set_description("Processing {:d} | {:d}...image base name: {}".format(image_id, 2*len(inds[split_flag]), base_name))
