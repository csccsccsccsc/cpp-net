# CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation

## Paper
The original paper of CPP-Net can be found in [TIP](https://ieeexplore.ieee.org/document/10024152) or [arxiv](https://arxiv.org/pdf/2102.06867.pdf).

## Requirements
```
    pytorch==1.11.0
    stardist==0.6.0
    csbdeep==0.6.3
```
Note that we only use several pre- or post-processing functions in StarDist.


## Prepare the datasets

```
Download DSB2018 used in stardist from https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
Download PanNuke from https://data.broadinstitute.org/bbbc/BBBC006 (In CPP-Net, we only use the "BBBC006_v1_images_z_16" images.)
Download PanNuke from https://jgamper.github.io/PanNukeDataset/
```

The details of separting the training / validation / test datasets can be found in reorganize_datasets/reorganize*.py.

Change "type_list" in the function getDataLoaders (in cppnet/dataloader_custom.py and feature_extractor/dataloader_aug.py) according to the names of your dataset splits (e.g., "split1" for training / "split2" for validation in PanNuke).


```
    DATA_PATH/train/images/*.tif
    DATA_PATH/val/images/*.tif
    DATA_PATH/test/images/*.tif
    DATA_PATH/train/masks/*.tif
    DATA_PATH/val/masks/*.tif
    DATA_PATH/test/masks/*.tif
    ...
```


## Prepare the instance shape-aware feature extractor

Modify the DATA_PATH in ./feature_extractor/main_shape.py

```
    python feature_extractor/main_shape.py --gpuid 0
```

## Train and Eval

Modify the SAP_Weight_path in ./cppnet/main_cppnet_dsb.py after the training process of SAP model

or set SAP_Weight_path=None to ignore the SAP Loss

Modify the DATA_PATH in ./cppnet/main_cppnet_dsb.py


```
    python cppnet/main_cppnet_dsb.py --gpuid 0
```

Modify the MODEL_WEIGHT_PATH in ./cppnet/main_cppnet_dsb.py after the training process of CPP-Net

Modify the DATASET_PATH_IMAGE and DATASET_PATH_LABEL in ./cppnet/main_cppnet_dsb.py
(e.g., DATASET_PATH_IMAGE=DATA_PATH/test/images and DATASET_PATH_LABEL=DATA_PATH/test/masks)



### Pytorch StarDist
There is a pytorch reimplementation of StarDist in https://github.com/ASHISRAVINDRAN/stardist_pytorch and part of the codes in our project are modified from this repository.
