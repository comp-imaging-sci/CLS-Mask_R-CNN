"""
Post-processing of predicted crown results
1) Generate 3D tif volume
2) 3D Label Map Interpoloation

Usage:
python post_processing_V2.py detection --logs_dir=/path/to/model/ --dataset_dir=/path/to/main_dataset/ --data_subset=sample_name --results_dir=/path/to/results/main_dataset/ --results_subset=sample_name_threshold

"""
import os
import sys
import re
import time
import random
import numpy as np
import tensorflow as tf
import skimage
from skimage.io import imsave, imread

lib_dir = '/home/xiaohui8/Desktop/crown_structure_seg/Mask_RCNN/mrcnn'
sys.path.append(lib_dir)

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_mono_mask
import mrcnn.model as modellib
from mrcnn.model import log
from detection import load_model, CrownDataset

def tiff_stack(dir_dataset):
    image_stack = []
    for image in os.listdir(dir_dataset):
        print(image)
        image_dir = os.path.join(dir_dataset, image)
        image_slice = imread(image_dir)
        skimage.external.tifffile.imsave(os.path.join(os.path.dirname(dir_dataset), '02-01.tif'), image_slice, append=True, compress=6)

def missing_label(set1, set2):
    """ Subtraction between two sets"""
    missing_list = list(set(set1) - set(set2))
    return missing_list

def layer_inspect(dataset, model, slice_index, anter_r, curr_r, iou_threshold):
    posterior_img_id = dataset.image_ids[slice_index]  
    posterior_img = dataset.load_ome_tif(posterior_img_id)
    posterior_r = model.detect([posterior_img], verbose=0)[0]  
               
    # Compute matches between slice n-1 and slice with "slice_index"
    _, ap_match, _ = utils.compute_matches(anter_r['rois'], anter_r['class_ids'], anter_r['masks'],
      posterior_r['rois'],  posterior_r['class_ids'], posterior_r['scores'], posterior_r['masks'], iou_threshold)
    
    # These crowns should appear in the current slice results
    ap_match_indices = np.where(ap_match > -1)

    # Compute matches between current slice and slice with "slice_index"
    _, cp_match, _ = utils.compute_matches(curr_r['rois'], curr_r['class_ids'], curr_r['masks'],
    posterior_r['rois'],  posterior_r['class_ids'], posterior_r['scores'], posterior_r['masks'], iou_threshold)

    cp_match_indices = np.where(cp_match > -1)
    missing_indices = np.setdiff1d(ap_match_indices, cp_match_indices)
    #missing_masks = anter_r['masks'][..., missing_indices]
    #curr_r['masks'] = np.append(curr_r['masks'], missing_masks, axis=2)
    
    return missing_indices, posterior_r

def append_missing_crowns(missing_indices, curr_r, posterior_r):
    # add rois
    missing_rois = posterior_r['rois'][missing_indices, ...]
    curr_r['rois'] = np.append(curr_r['rois'], missing_rois, axis = 0)
    # add class_ids
    missing_class_ids = posterior_r['class_ids'][missing_indices]
    curr_r['class_ids'] = np.append(curr_r['class_ids'], missing_class_ids, axis = 0)
    # add masks
    missing_masks = posterior_r['masks'][..., missing_indices]
    curr_r['masks'] = np.append(curr_r['masks'], missing_masks, axis=2)
    # 
    return curr_r

def label_map_interp(logs_dir, dataset_dir, data_subset, results_dir, results_subset):
    """ Compute IoU between several adjacent slices and add missing labels predicted by Mask R-CNN
    Inputs:
    logs_dir: directory to saved model
    dataset_dir: top directory of data to be predicted
    data_subset: sub-folder of data to be predicted
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

    # Compute interpolation starting from second slice and ended one slice before the last
    label_list = os.listdir(os.path.join(dataset_dir, data_subset))
    num_slices = len(label_list)

    # Define IOU threshold here!!!
    iou_threshold = 0.5

    # image slice 1
    anter_img_id = dataset.image_ids[0]
    anter_img = dataset.load_ome_tif(anter_img_id)
    anter_r = model.detect([anter_img], verbose=0)[0]
    print()
    # save result for the first image since it will be used in the inspection loop 
    anter_label_map = np.sum(anter_r['masks'], axis=2)
    skimage.io.imsave("{}/{}_pred.png".format(results_dir, dataset.image_info[anter_img_id]["id"]), anter_label_map)
    
    # save results of last few images since they will be ignored in the inspection loop 
    for m in range(num_slices-4, num_slices):
        last_img_id = dataset.image_ids[m]
        last_img = dataset.load_ome_tif(last_img_id)
        last_r = model.detect([last_img], verbose=0)[0]

        last_label_map = np.sum(last_r['masks'], axis=2)
        skimage.io.imsave("{}/{}_pred.png".format(results_dir, dataset.image_info[last_img_id]["id"]), last_label_map)

    for n in range(1, num_slices-4):
        print(n)
        curr_img_id = dataset.image_ids[n]
        curr_img = dataset.load_ome_tif(curr_img_id)
        curr_r = model.detect([curr_img], verbose=0)[0]
        print("Evaluating on center image:{}".format(dataset.image_info[curr_img_id]["id"]))

        missing_indices, posterior_r = layer_inspect(dataset, model, n+1, anter_r, curr_r, iou_threshold)
        if missing_indices.size == 0:
            missing_indices, posterior_r = layer_inspect(dataset, model, n+2, anter_r, curr_r, iou_threshold)
            if missing_indices.size == 0:
                missing_indices, posterior_r = layer_inspect(dataset, model, n+3, anter_r, curr_r, iou_threshold)
                if missing_indices.size == 0:
                    missing_indices, posterior_r = layer_inspect(dataset, model, n+4, anter_r, curr_r, iou_threshold)
                    curr_r = append_missing_crowns(missing_indices, curr_r, posterior_r)
                else:
                    curr_r = append_missing_crowns(missing_indices, curr_r, posterior_r)
            else:
                curr_r = append_missing_crowns(missing_indices, curr_r, posterior_r)         
        else:
            curr_r = append_missing_crowns(missing_indices, curr_r, posterior_r)

        # Update the corrected label map to anterior image predictions
        anter_r = curr_r      
        mono_color_map = np.sum(curr_r['masks'], axis=2)
        skimage.io.imsave("{}/{}_pred.png".format(results_dir, dataset.image_info[curr_img_id]["id"]), mono_color_map)
        #print("Finished and saved prediction for {}".format(dataset.image_info[image_id]["id"]))


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Crown Detection')
    parser.add_argument("command",
                        metavar="<command>",
                        help="label_map_interp")
    parser.add_argument('--logs_dir', required=False,
                        metavar="/path/to/model/",
                        help='Path to model')
    parser.add_argument('--dataset_dir', required=True,
                        metavar="/path/to/main_dataset",
                        help="Path to main dataset")
    parser.add_argument('--data_subset', required=False,
                        metavar="sample_name",
                        help='Path to specific sample')
    parser.add_argument('--results_dir', required=False,
                        metavar="/path/to/results/main_dataset",
                        help="Path to results")
    parser.add_argument('--results_subset', required=False,
                        metavar="sample_name_threshold",
                        help="Path to results subset")

    args = parser.parse_args()

    print("Dataset: ", args.dataset_dir)
    print("Subset: ", args.data_subset)

    if args.command == "detection":
        label_map_interp(args.logs_dir, args.dataset_dir, args.data_subset, args.results_dir, args.results_subset)

# ###################################
# Executables
# ###################################

#dataset_dir = '/home/xiaohui8/Desktop/'
#data_subset = 'test'

#logs_dir = '/home/xiaohui8/Desktop/logs_obese_all_data'
#dataset_dir = '/shared/anastasio1/SOMS/crown/raw/OA-RGND-8bit'
#data_subset = '00-00'
#results_dir = '/home/xiaohui8/Desktop/OA-RGND-8bit'
#results_subset = '00-00-0.85'

#tiff_stack('/home/xiaohui8/Desktop/OD-RPR-8bit/02-01-test')
#label_map_interp(logs_dir, dataset_dir, data_subset, results_dir, results_subset)
