# CLS-Mask_R-CNN
Crown-like Structure Detection and Segmentation in 3D Light-sheet Microscopy Imaging
This is an implementation of adapting Mask R-CNN of crown-like structure (CLS) detection and segmentation in light-sheet microscopy imaging.

The repository includes:
* Source code of adapting Mask R-CNN on CLS detection and segmentation
* Code for post-processing steps including slice compensation for maintaining 3D consistency of CLS and delineation of 3D CLS for the convenience of counting.
* Code for plotting free-response operation charactieristc (FROC) curve for performanace evaluation.
* Pre-trained model for CLS light-sheet microscopy images with ResNet-101 backbone
![pipeline](figures/pipeline.png)
