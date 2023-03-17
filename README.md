# CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation

## Requirements
```
    pytorch==1.11.0
    stardist==0.6.0
    csbdeep==0.6.3
```
Note that we only use several pre- or post-processing functions in StarDist.


## Prepare the related datasets
```
    DATA_PATH/train/images
    DATA_PATH/val/images
    DATA_PATH/test/images
    DATA_PATH/train/masks
    DATA_PATH/val/masks
    DATA_PATH/test/masks
```



## Prepare the instance shape-aware feature extractor

Modify the DATA_PATH in ./feature_extractor/main_shape.py

```
    python main_shape.py --gpuid 0
```

## Train and Eval

Modify the SAP_Weight_path in ./cppnet/main_cppnet_dsb.py after the training process of SAP model

Modify the DATA_PATH in ./cppnet/main_cppnet_dsb.py

```
    python main_cppnet_dsb.py --gpuid 0
```

Modify the MODEL_WEIGHT_PATH in ./cppnet/main_cppnet_dsb.py after the training process of CPP-Net

Modify the DATASET_PATH_IMAGE and DATASET_PATH_LABEL in ./cppnet/main_cppnet_dsb.py

(e.g., DATASET_PATH_IMAGE=DATA_PATH/test/images and DATASET_PATH_LABEL=DATA_PATH/test/masks)



### Pytorch StarDist
There is a pytorch reimplementation of StarDist in https://github.com/ASHISRAVINDRAN/stardist_pytorch and part of the codes in our project are modified from this repository.