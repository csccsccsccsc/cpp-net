# CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation

## Requirements
```
    pytorch==1.11.0
    stardist==0.6.0
    csbdeep==0.6.3
```


## Prepare the datasets
```
    DATA_PATH/train/images/*.tif or *.png
    DATA_PATH/val/images/*.tif or *.png
    DATA_PATH/test/images/*.tif or *.png
    DATA_PATH/train/masks/*.tif
    DATA_PATH/val/masks/*.tif
    DATA_PATH/test/masks/*.tif
    ...
```

Change the path in the script in reorganize_datasets, and run the script.
```
    python reorganize_datasets/reorganize_dsb2018.py
    python reorganize_datasets/reorganize_bbbc006.py
    python reorganize_datasets/reorganize_pannuke.py
```
The download link can also be found in these scripts.

Change "type_list" in the function getDataLoaders (in cppnet/dataloader_custom.py and feature_extractor/dataloader_aug.py) according to the names of your dataset splits.

## Prepare the instance shape-aware feature extractor

Modify the DATA_PATH in ./feature_extractor/main_shape.py. Here, the parameter --n_cls includes both foreground classes and the background.
Run the script like
```
    python feature_extractor/main_shape.py --gpuid 0 --dataset DSB2018 --n_cls 1
    python feature_extractor/main_shape.py --gpuid 0 --dataset PanNuke --n_cls 6
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

Modify the path in cppnet/predict_eval.p, and run the script to evaluate model performances

```
    python cppnet/predict_eval.py --gpuid 0
```

For each fold in PanNuke, use script cppnet/predict_eval_pannuke.py, and you can obtain a '.npy' file that includes predictions


### Pytorch StarDist
There is a pytorch reimplementation of StarDist in https://github.com/ASHISRAVINDRAN/stardist_pytorch and part of the codes in our project are modified from this repository.
