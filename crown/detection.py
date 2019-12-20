""" Detection pipeline with raw images
1) Change the 1-channel raw image to 3-channel
2) Detection
3) Save the binary masks
4) Connect crowns with high prediction scores in adjacent images
5) Construct 3D volume lightsheet image 
"""

import os
import sys
import re
import time
import random
import numpy as np
import tensorflow as tf
import skimage
from skimage import img_as_ubyte
from skimage.measure import label
from skimage.color import label2rgb
from skimage.io import imread, imsave

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

lib_dir = '/home/xiaohui8/Desktop/crown_structure_seg/Mask_RCNN/mrcnn'
sys.path.append(lib_dir)

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_mono_mask
import mrcnn.model as modellib
from mrcnn.model import log

# #######################
# Configuration
# #######################

class CrownInferenceConfig(Config):
    """Configuration for inference on the crown segmentation dataset."""

    NAME = "crown"
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus
	# Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    # ROIs kept after non-maximum supression (inference)
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_NMS_THRESHOLD = 0.7

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 250

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 250

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.95

    DETECTION_NMS_THRESHOLD = 0.5

    # Supported values are: resnet50, resnet101
    # BACKBONE = "resnet101"
    BACKBONE = "resnet101"

    # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_SCALES = (16, 32 ,64 ,128 ,256)

    LEARNING_RATE = 0.001


# #######################
# Dataset
# #######################

class CrownDataset(utils.Dataset):

    def triple_channel(self, dataset_dir, temp_dataset_dir, subset, num_channels=3):
        
        dataset_dir = os.path.join(dataset_dir, subset)
        temp_dataset_dir = os.path.join(temp_dataset_dir, subset)

        if not os.path.exists(temp_dataset_dir):
            os.makedirs(temp_dataset_dir)        
 
        for image_id in os.listdir(dataset_dir):
            image_filename = os.path.join(dataset_dir, image_id)

            # When read ome.xml file, set "is_ome = False" to disable reading metadata
            image = skimage.external.tifffile.imread(image_filename, is_ome = False) 
            new_image = np.repeat(image[..., np.newaxis], num_channels, -1)
            skimage.external.tifffile.imsave(os.path.join(temp_dataset_dir, image_id), new_image)

            print("Sucessfully saved image {} in 3-channel format").format(images)

    def file_base_name(self, file_name):
        """ Handle file name with multiple '.' and get the prefix
        """ 
        if '.' in file_name:
            separator_index = file_name.index('.')
            base_name = file_name[:separator_index]
            return base_name
        else:
            return file_name

    def path_base_name(self, path):
        file_name = os.path.basename(path)
        return self.file_base_name(file_name) 

    def load_crown(self, dataset_dir, subset):
          """ Load images to predict

          dataset_dir: root directory of dataset 
          subset: name of the subset to run the prediction
          """
          self.add_class("crown", 1, "crown")      
          dataset_dir = os.path.join(dataset_dir, subset)    
          if not os.path.exists(dataset_dir):
              print("No directory: {} exists".format(dataset_dir))
          
          # Walk down the images in the directory
          # image_ids = [self.path_base_name(file) for file in os.listdir(dataset_dir)]
          image_ids = [self.path_base_name(file) for file in sorted(os.listdir(dataset_dir))]
          
          # Add images to the dataset
          for image_id in image_ids:
              self.add_image("crown", image_id=image_id,
                       path = os.path.join(dataset_dir, "{}.ome.tif".format(image_id)))

    def load_mask(self, image_id):
        """ Load mask dataset
        Mask directory needs to be specify
        """
        info = self.image_info[image_id]
        # Specify directory for mask datset
        mask_dir = os.path.join("/shared/curie/SOMS/crown/mrcnn/raw-combine",info['id'], "masks")    
        mask = []
   
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)

        mask = np.stack(mask, axis=-1)
        
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
   
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "crown":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


    # @property
    def load_ome_tif(self, image_id):
        """Load the ome.tif lightsheet image and return a [H,W,3] Numpy array.
        """
        # Load ome.tif image
        image = skimage.external.tifffile.imread(self.image_info[image_id]['path'], is_ome = False)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image



def load_model(logs_dir):
    
    config = CrownInferenceConfig()
    # Device (use "/cpu:0" if running on local cpu)
    DEVICE = "/gpu:0"
    with tf.device(DEVICE):
       model = modellib.MaskRCNN(mode ="inference", model_dir=logs_dir, config=config)
    
    # Find latest model and load weights
    # weights_path = model.find_last()
    # weights_path = '/home/xiaohui8/Desktop/best_model/mask_rcnn_crown_0039.h5'
    weights_path = '/home/xiaohui8/Desktop/logs_obese_all_data/crown20190907T1704/mask_rcnn_crown_0063.h5'
    model.load_weights(weights_path, by_name=True)   
    
    return model, config

def label_image(gt_dir, pred_dir, difference_dir):

    if os.path.exists(gt_dir):
        gt = os.listdir(gt_dir)        
        gt.sort()

    if os.path.exists(pred_dir):
       pred = os.listdir(pred_dir)        
       pred.sort()
 
     # index = 1 # Initialize Ztag counting

    # prefix[0]: filename,  prefix[1]:extension
    for index in range(1, 919):
         print(index)
         Ztag4 = 'Z'+ str(index).zfill(4)
         Ztag3 = 'Z' +str(index).zfill(3)
         for file in gt:
             if file.endswith(Ztag3 + ".tif"):
                 gt_image = imread(os.path.join(gt_dir, file))
                 gt_label = label(gt_image, return_num=True)
                 # Ground truth mask is green
                 gt_colors = [(0, 1, 0)] * gt_label[1]
                           
                 gt_label_color = label2rgb(gt_label[0], bg_label=0, bg_color=(0, 0, 0), colors = gt_colors)
                 
         for file in pred:
             if file.endswith(Ztag4 + "_pred.png"):
                 pred_img = imread(os.path.join(pred_dir, file))
                 pred_label = label(pred_img, return_num = True)
                 # Predicted mask is red
                 pred_colors = [(1, 0 ,0)] * pred_label[1] 
                 pred_label_color = label2rgb(pred_label[0], bg_label=0, bg_color=(0, 0, 0), colors = pred_colors, image = gt_label_color, kind = "overlay")

         fig, ax = plt.subplots(figsize=(16,16))
         #  ax.imshow(pred_label_color)
         #  plt.show()
         imsave(os.path.join(difference_dir, Ztag3 + ".png"), pred_label_color)
         print("save image: {}".format(os.path.join(pred_dir, file)))

    

def detect_false_positives(logs_dir, dataset_dir, data_subset, results_dir, results_subset):
    """Run detection on images and compute false positives and save mono-color false positives map
    Inputs:
    logs_dir: can be initialized with any random path for now
    dataset_dir: root directory of images where detection runs
    data_subset: folder in the root directory where detection runs
    results_dir: root directory of images where results will be saved
    results_subset: folder in the root directory where results will be saved
    """
    # Load model, dataset and network configuration
    model, config = load_model(logs_dir)
    dataset = CrownDataset()
    dataset.load_crown(dataset_dir, data_subset)
    dataset.prepare()
    results_dir = os.path.join(results_dir, results_subset)

    # Create a directory to store the results if not given
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("Created directory {} to save results".format(results_dir))

    for image_id in dataset.image_ids:
        # Load image and run detection
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        print("Reading image {}".format())

        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=0)
        r = results[0]

        # Match betweeen the predicted labels and ground truth labels
        gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
        r['rois'],  r['class_ids'], r['scores'], r['masks'], iou_threshold = 0.5)
        # False positives(prediton with no matched ground truth labels)
        pred_match_indices = np.where(pred_match == -1)
        image = dataset.load_ome_tif(image_id)
        r_temp = model.detect([image], verbose=0)[0]
        pred_match_indices = np.array(pred_match_indices)
        pred_match_indices = np.array(pred_match_indices[np.where(pred_match_indices<(r_temp['masks'].shape)[2])])
        
        fp_mask = r_temp['masks'][..., pred_match_indices]
        fp_mask = np.sum(np.squeeze(fp_mask), axis=2) 
        # Save label map      
        skimage.io.imsave("{}/{}_fp.png".format(results_dir, dataset.image_info[image_id]["id"]), fp_mask.astype(np.int8))

        print("Saved false positives for {}".format(dataset.image_info[image_id]["id"]))

def detect(logs_dir, dataset_dir, data_subset, results_dir, results_subset):
    """ Run detection only and save mono-color label maps
    Inputs:
    logs_dir: can be initialized with any random path for now
    dataset_dir: root directory of images where detection runs
    data_subset: folder in the root directory where detection runs
    results_dir: root directory of images where results will be saved
    results_subset: folder in the root directory where results will be saved
    """
    # Load model, dataset and network configuration
    model, config = load_model(logs_dir)
    dataset = CrownDataset()
    dataset.load_crown(dataset_dir, data_subset)
    dataset.prepare()
    results_dir = os.path.join(results_dir, results_subset)    

    # Create a directory to store the results if not given
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("Created directory:{} to save results".format(results_dir))
 
    for image_id in dataset.image_ids:
        # Load image
        image = dataset.load_ome_tif(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0] 
        num_crown = r["masks"].shape[2]
        print("Number of crown detected is {}".format(num_crown))
       
        if num_crown == 0:
            # If no crown structures are detected, save label map as all-zeros map
            
            label_map = np.zeros((r["masks"].shape[0], r["masks"].shape[1]))
        else:
            # Combine the seperated detected instance masks
            label_map = np.sum(r["masks"], axis=2)

        # Save label map
        skimage.io.imsave("{}/{}_pred.png".format(results_dir, dataset.image_info[image_id]["id"]), label_map)
        print("Finished and saved prediction for {}".format(dataset.image_info[image_id]["id"]))

    print("Crown detection finished for {}".format(os.path.basename(dataset_dir)))       



# ####################################
# Executables
# ####################################
#logs_dir = '/home/xiaohui8/Desktop/logs_obese_all_data'
#dataset_dir = '/home/xiaohui8/Desktop/'
#data_subset = 'test'
#results_dir = '/home/xiaohui8/Desktop'
#results_subset = 'result'
#dataset_dir = '/shared/curie/SOMS/crown/raw/OD-LPR-8bit'
#data_subset = '02-01'
#dataset_dir = '/home/xiaohui8/Desktop'
#data_subset = 'test'
#results_dir = '/home/xiaohui8/Desktop'
#results_subset = '02-01'
#detect(logs_dir, dataset_dir, data_subset, results_dir, results_subset)
#detect_false_positives(logs_dir, dataset_dir, data_subset, results_dir, results_subset)
#label_image('/shared/curie/SOMS/crown/raw/OE_LPR_01-01/Segmentation', '/home/xiaohui8/Desktop/01-01-pred', '/home/xiaohui8/Desktop/01-01-difference')
