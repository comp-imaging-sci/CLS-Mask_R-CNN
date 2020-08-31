# Mask R-CNN for Crown-like Structure Detection and Segmentation in 3D Light-sheet Microscopy Imaging
This is an implementation of adapting Mask R-CNN of crown-like structure (CLS) detection and segmentation in light-sheet microscopy imaging.
<p align="center">
    <img src="figures/overlay_segmentation.png" width="324" height="324">
</p>
<p align="center">
    <b><em>Visualization of segemnted CLS masks overlaid on nucleus image</em></b>
</p>

The repository includes:
* Source code of adapting Mask R-CNN on CLS detection and segmentation
* Code for post-processing steps including slice compensation for maintaining 3D consistency of CLS and delineation of 3D CLS for the convenience of counting.
* Code for plotting free-response operation charactieristc (FROC) curve for performanace evaluation.
* Pre-trained model for CLS light-sheet microscopy images with ResNet-101 backbone
- Mask R-CNN architecure
<p align="center">
    <img src="figures/maskrcnn.png" width="600" height="300">
</p>

- CLS detection and segmentation pipeline visulaization 
![pipeline](figures/pipeline.png)

- Post-processing results to maintain 3D CLS structure and label delineation for counting
<p align="center">
    <img src="figures/post_processing.png" width="600" height="200">
</p>

This code is an extension from the work of [matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN). Please condider to cite both repositories (blbbex below) if you are using this in your research. You can download the pre-trained weights on [Google Drive Pretrain Mask R-CNN CLS](https://drive.google.com/file/d/10vgXowD2M8xRrs6-A5pXCUbDlOUfan2A/view?usp=sharing) and put in under this directory and change the path to load the model.

## Requirements
```
- Python 3.7.3
- Tensorflow 1.13
- Keras 2.2
- Other packages listed in the requirements.txt
```
## Implementation
### 1. Data Preparation and Pre-processing
#### Training Your Own Model
The original image data should seperate all raw light-sheet microscopy images and corresponding human annotated masks in two individual folders. By running the function `directory_organizer` in `pre_processing.py`, all training data should be organized as the file structure shown below:
```
image_name
├── images
│   └── image_name.tif
└── masks
    ├── CLS_mask1.png
    ├── CLS_mask2.png
    ├── CLS_mask3.png
```
#### Using Our Pretrained Model to Detect CLS
Just simply put all raw images in one folder

### 2. Detect CLS Using Our Pretrained Model
```
./run_detection.sh detection --logs_dir=/path/to/pretrain/model/ --dataset_dir=/path/to/dataset/ --data_subset=/path/to/subfolder/in/dataset/folder --results_dir=/path/to/save/results --results_subset=/path/to/subfolder/in/results/folder
```
## Citations
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
