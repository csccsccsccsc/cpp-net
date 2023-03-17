import os
import numpy as np
from skimage import io
import warnings
warnings.filterwarnings("ignore")

#######
### 1) Download PanNuke from https://jgamper.github.io/PanNukeDataset/
### 2) Save each image and mask in the origianl ".npy" file as ".tif"

for ifold in [1,2,3]:

    image_dir = 'DATA ROOT PATH/pannuke/fold_'+str(ifold)+'/images/fold'+str(ifold)+'/images.npy'
    masks_dir = 'DATA ROOT PATH/pannuke/fold_'+str(ifold)+'/masks/fold'+str(ifold)+'/masks.npy'
    imgs = np.load(image_dir).astype(np.uint8)
    msks = np.load(masks_dir).astype(np.uint16)

    img_filefold_tosave = os.path.join(os.getcwd(), 'reorganized_dataset', 'fold_'+str(ifold), 'images')
    msk_filefold_tosave = os.path.join(os.getcwd(), 'reorganized_dataset', 'fold_'+str(ifold), 'masks')

    if not(os.path.exists(img_filefold_tosave)):
        os.makedirs(img_filefold_tosave)
    if not(os.path.exists(msk_filefold_tosave)):
        os.makedirs(msk_filefold_tosave)
    img_name_list_file = open(os.path.join(img_filefold_tosave, 'name_list.txt'), 'w')
    msk_name_list_file = [ open(os.path.join(msk_filefold_tosave, 'name_list_c'+str(imod)+'.txt'), 'w') for imod in range(6) ]

    nimg = imgs.shape[0]

    for iimg in range(nimg):
        io.imsave(os.path.join(img_filefold_tosave, 'img_'+str(iimg)+'.png'), imgs[iimg])
        img_name_list_file.write(str(iimg) + ',' + os.path.join(img_filefold_tosave, 'img_'+str(iimg)+'.png') + '\n')
        print(str(iimg) + ',' + os.path.join(img_filefold_tosave, 'img_'+str(iimg)+'.png'))
        for imod in range(6):
            io.imsave(os.path.join(msk_filefold_tosave, 'msk_'+str(iimg)+'_c_'+str(imod)+'.tif'), msks[iimg, :, :, imod])
            msk_name_list_file[imod].write(str(iimg) + ',' + os.path.join(msk_filefold_tosave, 'msk_'+str(iimg)+'_c_'+str(imod)+'.tif') + '\n')

    img_name_list_file.close()
    for imod in range(6):
        msk_name_list_file[imod].close()
