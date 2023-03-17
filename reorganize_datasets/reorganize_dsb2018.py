import os
from shutil import copyfile
from skimage import io
import numpy as np

#######
### 1) Download DSB2018 used in stardist from https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
### 2) Split the training dataset into sub-training and sub-validation sets (random state = 42)

root_dir = 'DATA ROOT PATH/dsb2018_in_stardist/dsb2018/train'
save_root_dir = 'DATA ROOT PATH/dsb2018_in_stardist/dsb2018/dataset_split_for_training'
name_list = sorted(os.listdir(os.path.join(root_dir, 'images')))

rng = np.random.RandomState(42)
ind = rng.permutation(len(name_list))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
print('number of images: %3d' % len(name_list))
print('- training:       %3d' % len(ind_train))
print('- validation:     %3d' % len(ind_val))

for flag in ['train', 'val']:
    if not(os.path.exists(os.path.join(save_root_dir, flag, 'images'))):
        os.makedirs(os.path.join(save_root_dir, flag, 'images'))
    if not(os.path.exists(os.path.join(save_root_dir, flag, 'masks'))):
        os.makedirs(os.path.join(save_root_dir, flag, 'masks'))
        
with open(os.path.join(save_root_dir, 'train_name.txt'), 'a') as f:
    for image_id, name in enumerate(name_list):
        if image_id in ind_val:
            split_flag = 'val'
        elif image_id in ind_train:
            split_flag = 'train'
        else:
            print('Not working')
            
        # Copy Image
        img_source = os.path.join(root_dir, 'images', name)
        img = io.imread(img_source)
        img_target = os.path.join(save_root_dir, split_flag, 'images', str(image_id)+'.tif')
        io.imsave(img_target, img)
           
        # Copy Image
        mask_source = os.path.join(root_dir, 'masks', name)
        mask = io.imread(mask_source)
        mask_target = os.path.join(save_root_dir, split_flag, 'masks', str(image_id)+'.tif')
        io.imsave(mask_target, mask)
        print(image_id, img.dtype, mask.dtype, img.shape, mask.shape)

        
        # if len(mask_name_list) >= 256:
            # print(image_id, len(mask_name_list), mask.max(), mask[mask>0].min())

        # Filename record
        f.write(str(image_id)+','+name+','+split_flag+'\n')