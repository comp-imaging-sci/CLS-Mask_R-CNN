"""
 Implementation of  FROC Analysis on Mask R-CNN
 "Maximum Likelihood Analysis of Free-Response receiver operating characteristic (FROC) data", Dev P. Chakraborty, 1998
 Xiaohui Zhang
 Jul 25, 2019
"""

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
ROOT_DIR = '/home/xiaohui8/Desktop'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import crown_v2

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs_obese_all_data")
print(LOGS_DIR)

# ########################
# Configurations
# ########################

# Dataset directory
DATASET_DIR = '/shared/anastasio1/SOMS/crown/mrcnn/all_data'
DEVICE = "/gpu:2" # /cpu:0
TEST_MODE = "inference"

# ##########################
#  Load Validation Dataset
# ##########################

dataset = crown_v2.NucleusDataset()
dataset.load_crown(DATASET_DIR, "all_data_test")
dataset.prepare()

# #########################
# FROC Analysis
# #########################
confidence_range = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
mean_TPs = []
mean_FPs = []
TP_fractions = []
for confidence in confidence_range:
    print("Running evaluation of confidence threshold {}".format(confidence))
    
    config = crown_v2.NucleusInferenceConfig()
    config.DETECTION_MIN_CONFIDENCE = confidence
    
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)
    
    # Load the last model you trained and the weights
    #weights_path = model.find_last()
    weights_path = '/home/xiaohui8/Desktop/logs_obese_all_data/crown20190907T1704/mask_rcnn_crown_0063.h5'
    #weights_path = '/home/xiaohui8/Desktop/best_model/mask_rcnn_crown_0039.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    
    # Load validation dataset
    image_ids = dataset.image_ids
    
    # Mean number of FPs per images, TP fraction, Total preditions
    FPs = []
    TPs = []
    FNs = []
    
   
    for image_id in image_ids:
        
        print(dataset.image_info[image_id])
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        
        info = dataset.image_info[image_id]
        
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
        r = results[0]
        
        gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
        r['rois'],  r['class_ids'], r['scores'], r['masks'], iou_threshold = 0.5)

        num_total_gt = gt_bbox.shape[0]
        num_total_pred = r['rois'].shape[0]
        
        try:       
            num_TP = np.cumsum(pred_match > -1)[-1]
        except IndexError:
            num_TP = 0

        num_FN = gt_bbox.shape[0] - num_TP
        num_FP = num_total_pred - num_TP
        
        print("Finished one images")
        print("num_TP:", num_TP)
        print("num_FN:", num_FN)
        print("num_FP:", num_FP)
       
        TPs.append(num_TP)
        FPs.append(num_FP)   
        FNs.append(num_FN)

    mean_FP = np.mean(FPs)
    TP_fraction = np.sum(TPs)/ (np.sum(TPs) + np.sum(FNs))

    mean_FPs.append(mean_FP)
    TP_fractions.append(TP_fraction)
    
    mean_TP = np.mean(TPs)
    mean_TPs.append(mean_TP)
     
    sio.savemat('/home/xiaohui8/Desktop/crown_structure_seg/crown_seg_results/statistics/FROC_mat/all_data_0101_epoch63_FP_extra.mat', {'mean_FPs':mean_FPs})
    sio.savemat('/home/xiaohui8/Desktop/crown_structure_seg/crown_seg_results/statistics/FROC_mat/all_data_0101_epoch63_TP_fraction_extra.mat', {'TP_fractions':TP_fractions})
    sio.savemat('/home/xiaohui8/Desktop/crown_structure_seg/crown_seg_results/statistics/FROC_mat/all_data_0101_epoch63_TP_extra.mat', {'mean_TPs':mean_TPs})
print(mean_TPs)
print(mean_FPs)
print(TP_fractions)
    


