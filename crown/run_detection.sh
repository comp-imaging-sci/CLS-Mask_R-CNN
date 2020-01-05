#!/bin/bash 
KERAS_BACKEND=tensorflow python /home/xiaohui8/Desktop/crown_structure_seg/Mask_RCNN/samples/crown/post_processing_V2.py "$@"

# add arguments of "detection --logs_dir=/path/to/pretrain/model/ --dataset_dir=/path/to/dataset/ 
# --data_subset=/path/to/subfolder/in/dataset/folder --results_dir=/path/to/save/results 
# --results_subset=/path/to/subfolder/in/results/folder" when running ./run_detection
