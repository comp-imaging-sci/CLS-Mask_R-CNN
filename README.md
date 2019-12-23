# Mask R-CNN for Crown-like Structure Detection and Segmentation in 3D Light-sheet Microscopy Imaging
This is an implementation of adapting Mask R-CNN of crown-like structure (CLS) detection and segmentation in light-sheet microscopy imaging.

The repository includes:
* Source code of adapting Mask R-CNN on CLS detection and segmentation
* Code for post-processing steps including slice compensation for maintaining 3D consistency of CLS and delineation of 3D CLS for the convenience of counting.
* Code for plotting free-response operation charactieristc (FROC) curve for performanace evaluation.
* Pre-trained model for CLS light-sheet microscopy images with ResNet-101 backbone
- Mask R-CNN architecure
![maskrcnn](figures/maskrcnn.png)

- CLS detection and segmentation pipeline visulaization 
![pipeline](figures/pipeline.png)

This code is an extension from the work of [matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN). Please condider to cite both repositories (blbbex below) if you are using this in your research. You can download the pre-trained weights on [Googl Drive Pretrain Mask R-CNN CLS](https://drive.google.com/open?id=10vgXowD2M8xRrs6-A5pXCUbDlOUfan2A) and put in under this directory and change the path to load the model.

## Requirements
## Implementation
## Citations
