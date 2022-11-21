# Installation & Data Preparetion

## Requirements

* Linux
* Python>=3.6.2 and < 3.9
* PyTorch>=1.4
* torchvision (matching PyTorch install)
* CUDA (must be a version supported by the pytorch version)
* OpenCV (optional)

## Installing FALSE

1. Create Enviroment
```
conda create -n false_env python=3.8
conda activate false_env
```
2. Install PyTorch
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch
```
3. Install Apex(optional)
```
git clone --recursive https://www.github.com/NVIDIA/apex
cd apex
python3 setup.py install
```
4. Install FALSE
```
git clone --recursive https://github.com/GeoX-Lab/FALSE.git
cd FALSE
pip install --progress-bar off -r requirements.txt
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/master
pip install -e .[dev]
```

## Data Preparetion

1. Prepare Data, e.g.,
```
SegmentationImage
|_ <potsdam>
|   _ <ssl_train>
|  |  |_ <train>
|  |  |  |_ <img-t1-name>.tif
|  |  |  |_ ...
|  |  |  |_ <img-tN-name>.tif
|  |  |  |_ ...
|   _ <ssl_val>
|  |  |_ <val>
|  |  |  |_ <img-v1-name>.tif
|  |  |  |_ ...
|  |  |  |_ <img-vN-name>.tif
|  |  |  |_ ...
|_ <dglc>
|  |_ <ssl_train>
|  |  |_ <train>
|  |  |  |_ <img-t1-name>.tif
|  |  |  |_ ...
|  |  |  |_ <img-tN-name>.tif
|  |  |  |_ ...
|_ <xiangtan>
|  |_ <ssl_train>
|  |  |_ <train>
|  |  |  |_ <img-t1-name>.tif
|  |  |  |_ ...
|  |  |  |_ <img-tN-name>.tif
|  |  |  |_ ...
```
2. Set Data Path

move to [dataset_catalog.json](../configs/config/dataset_catalog.json) and add (e.g. Potsdam):
```
{
    "potsdam":{
        "train":["SegmentationImage/potsdam/ssl_train"," "],
        "val":["SegmentationImage/potsdam/ssl_val"," "]
    }
}