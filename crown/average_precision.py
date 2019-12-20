# Implementation of  FROC Analysis on Mask R-CNN
# "Maximum Likelihood Analysis of Free-Response receiver operating characteristic (FROC) data", Dev P. Chakraborty, 1998
# Xiaohui Zhang
# Jul 25, 2019

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio

# Root directory of the project
ROOT_DIR = '/home/xiaohui8/Desktop/crown_structure_seg/Mask_RCNN/'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import crown


# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs_obese_lean")
print(LOGS_DIR)

# ########################
# Configurations
# ########################

# Dataset directory
DATASET_DIR = '/shared/curie/SOMS/crown/mrcnn'
DEVICE = "/gpu:1" # /cpu:0
TEST_MODE = "inference"

# ##########################
#  Load Validation Dataset
# ##########################

# dataset = nucleus.NucleusDataset()
dataset = crown.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "val")
dataset.prepare()

# Inherit the config class
config = crown.NucleusInferenceConfig()
config.DETECTION_MAX_INSTANCES = 250
config.DETECTION_MIN_CONFIDENCE = 0.95
config.display()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)

# Load the last model you trained and the weights
weights_path = model.find_last()
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# Load validation dataset
image_ids = dataset.image_ids
 
# ##############################
# Compute Precision-Recall Curve
# ##############################
# test one image 
APs = []

for image_id in image_ids:
    # Load image
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

    # Run object detection
    results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
    # Compute AP over range 0.5 to 0.95
    r = results[0]

    iou_thresholds = np.arange(0.5, 1.0, 0.05, dtype = np.float32)
    root_save_dir = '/home/xiaohui8/Desktop/crown_structure_seg/AP_mat/'
 
    iou_threshold = 0.5

    ap, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, 
                 r['rois'],  r['class_ids'], r['scores'], r['masks'], iou_threshold = iou_threshold)
    
    filename_ap = os.path.join(root_save_dir, 'ap_iou_thres_' + str(iou_threshold) + '.mat')
    
    sio.savemat(filename_ap, {'ap':ap})

    filename_APs = os.path.join(root_save_dir, 'APs.mat')
    APs.append(ap)
   
    sio.savemat(filename_APs, {'APs': APs})       


