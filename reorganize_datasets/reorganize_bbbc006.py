import os
import glob
from shutil import copyfile
from skimage import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tqdm
import shutil

img_dir = './BBBC006_v1_images_z_16'
msk_dir = './BBBC006_v1_labels'
save_img_dir = './reorganize_dataset'

if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)    

img_name_list = os.listdir(os.path.join(img_dir))

name_list = []
msk_loc_list = {}
img_loc_list_w1 = {}
img_loc_list_w2 = {}
for name in img_name_list:
    name_terms = name.split('_')
    if name_terms[3][0:2] == 'w1':
        name_list.append('_'.join(name_terms[0:3]))
        img_loc_list_w1.update({'_'.join(name_terms[0:3]) : os.path.join(img_dir, name)})
        msk_loc_list.update({'_'.join(name_terms[0:3]) : os.path.join(msk_dir, '_'.join(name_terms[0:3])+'.png')})
    else:
        img_loc_list_w2.update({'_'.join(name_terms[0:3]) : os.path.join(img_dir, name)})

# split training, validation, and testing
# Note that each 'w1' and 'w2' image pair of ths same image is groupedd into the same dataset.
n_img = len(name_list)
rngstate = np.random.RandomState(42)
testset = rngstate.choice(n_img, int(0.2*n_img), replace=False)
trainvalset = np.setdiff1d(np.arange(n_img), testset)
valset = rngstate.choice(trainvalset, int(0.2*n_img), replace=False)
trainset = np.setdiff1d(trainvalset, valset)
print('total : ', 2*n_img)
print('train : ', 2*len(trainset))
print('val : ', 2*len(valset))
print('test : ', 2*len(testset))

dataset_idxlist = {'val':valset, 'test':testset, 'train':trainset}
for dataset_type in ['train', 'val', 'test']:
    cur_save_dir = os.path.join(save_img_dir, dataset_type)
    idxlist = dataset_idxlist[dataset_type]

    print('Processing '+dataset_type+' dataset (including '+str(2*len(idxlist))+' images')
    print(cur_save_dir)


    if not os.path.exists(os.path.join(cur_save_dir, 'w1_images')):
        os.makedirs(os.path.join(cur_save_dir, 'w1_images'))
    if not os.path.exists(os.path.join(cur_save_dir, 'w1_masks')):
        os.makedirs(os.path.join(cur_save_dir, 'w1_masks'))
    if not os.path.exists(os.path.join(cur_save_dir, 'w2_images')):
        os.makedirs(os.path.join(cur_save_dir, 'w2_images'))
    if not os.path.exists(os.path.join(cur_save_dir, 'w2_masks')):
        os.makedirs(os.path.join(cur_save_dir, 'w2_masks'))


    processed_list = []
    processed_idx_list = []
    f_img = open(os.path.join(cur_save_dir, dataset_type + '_images_name_list.txt'), 'w')
    f_msk = open(os.path.join(cur_save_dir, dataset_type + '_masks_name_list.txt'), 'w')
    for idx in idxlist:
        f_img.write(os.path.join(cur_save_dir, 'w1_images', name_list[idx]+'_w1.png') + '\n')
        f_msk.write(os.path.join(cur_save_dir, 'w1_masks', name_list[idx]+'_w1.png') + '\n')
        f_img.write(os.path.join(cur_save_dir, 'w2_images', name_list[idx]+'_w2.png') + '\n')
        f_msk.write(os.path.join(cur_save_dir, 'w2_masks', name_list[idx]+'_w2.png') + '\n')


        if not( os.path.join(cur_save_dir, 'w1_images', name_list[idx]+'_w1.png') in processed_list ):
            processed_list.append(os.path.join(cur_save_dir, 'w1_images', name_list[idx]+'_w1.png'))
            processed_idx_list.append(idx)
        else:
            print(os.path.join(cur_save_dir, 'w1_images', name_list[idx]+'_w1.png'))


        if not( os.path.join(cur_save_dir, 'w2_images', name_list[idx]+'_w2.png') in processed_list ):
            processed_list.append(os.path.join(cur_save_dir, 'w2_images', name_list[idx]+'_w2.png'))
        else:
            print(idx, os.path.join(cur_save_dir, 'w2_images', name_list[idx]+'_w2.png'))
            for pi, p in zip(processed_idx_list, processed_list):
                print(pi, p)
            assert(False)

        # NOTE: the original images are '.tif' files...
        shutil.copy(img_loc_list_w1[name_list[idx]], os.path.join(cur_save_dir, 'w1_images', name_list[idx]+'_w1.png'))
        shutil.copy(img_loc_list_w2[name_list[idx]], os.path.join(cur_save_dir, 'w2_images', name_list[idx]+'_w2.png'))
        shutil.copy(msk_loc_list[name_list[idx]], os.path.join(cur_save_dir, 'w1_masks', name_list[idx]+'_w1.png'))
        shutil.copy(msk_loc_list[name_list[idx]], os.path.join(cur_save_dir, 'w2_masks', name_list[idx]+'_w2.png'))
