"""
Xiaohui Zhang
Jan rth, 2020
1) Check all images and remove those with no crowns in it 
2) Create Instance Mask for MASK R-CNN Implementation
3) Organize the original dataset 
4) Convert 1-channel image to 3-channel RGB-like image
5) Convert 16-bit to 8-bit image
6) View the image

"""
import sys
import os
from os.path import isfile, join, splitext
import shutil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

import skimage
from skimage import img_as_uint, img_as_ubyte
from skimage.external.tifffile import imread, imsave, imshow
from skimage.measure import label, regionprops

def generate_instance_mask(binary_mask, masks_dir):
    """
    Seperate the crown structure labels in a binary mask
    Input:
    binary mask: numpy array of binary masks
    masks_dir: directory to save the seperated instance masks
    """
    # Label the binary label mask
    label_img = label(binary_mask)
    extension = '.png'

    print("generating masks...")
    for instance in regionprops(label_img):

        # Seperate the crown struture labels according to the size
        if instance.area > 100:
            index_mask = instance.label * np.ones(binary_mask.shape)
            instance_mask = (index_mask == label_img)

            # Save each crown label as a binary map seperately 
            save_filename = join(masks_dir, str(instance.label) + extension)
            skimage.io.imsave(save_filename, img_as_ubyte(instance_mask))


def remove_all_zeros_lean(folder_to_clean_dir):
    """
    Remove lean mice cases with no crown structures label 
    """
    img_list = next(os.walk(folder_to_clean_dir))[2]
    parent_dir = os.path.abspath(join(folder_to_clean_dir, os.pardir))
    
    # Walk through every label map in the "segmentation" folder 
    for img_file in img_list:

        prefix = os.path.splitext(img_file)[0]
        extension = os.path.splitext(img_file)[1]

        img_fullpath = join(next(os.walk(folder_to_clean_dir))[0], img_file)
        img = imread(img_fullpath)

        if np.count_nonzero(img) == 0:
            # Reomve the label file with no crown appearing
            os.remove(img_fullpath)
          
            # Remove the corresponding original image file to the removed label file          
            for file in os.listdir(join(parent_dir, 'deconv')):                
                if file.endswith(prefix[-3:] + '.tif'):
                    os.remove(join(parent_dir, 'deconv', file))
                    print('Removed original image: {}').format(join(parent_dir, 'deconv', file))       
      
            print('Removed label mask file: {}').format(img_fullpath)


def directory_organizer(root_dir_images, root_dir_masks, dir_to_save):
    """
    Operation of organizing directory as the format of nuclei segmentation challenge

    Input: 
    root_dir_images: the root folder directory that contains all training images
    root_dir_masks: the root filder directory that contains all training labels 
    dir_to_save: the root directory where all the new subset folders are created
    """
    img_prefix = []
    extension = 'ome.tif'

    if os.path.exists(root_dir_images):
        img_prefix = [os.path.splitext(file) for file in os.listdir(root_dir_images)]        
        img_prefix.sort()
        
    index = 0 # Initialize Ztag counting

    # prefix[0]: filename,  prefix[1]:extension
    for prefix in img_prefix:
         
         Ztag4 = 'Z'+ str(index).zfill(4)
        # Ztag3 = 'Z' +str(index).zfill(3)
         Ztag3 = str(index).zfill(3)
         
         print "creating directory for {}".format(prefix[0][:-4])
         new_dir = join(dir_to_save, prefix[0][:-4])
  
         if not os.path.exists(new_dir):
            os.makedirs(new_dir)

         images_dir = join(new_dir, 'images')
         masks_dir = join(new_dir, 'masks')
     
         if not os.path.exists(images_dir):
             os.makedirs(images_dir)       
                
             for file in os.listdir(root_dir_images):  
                 if file.endswith(Ztag4 + '.ome.tif'):
                     shutil.copy(join(root_dir_images, file), images_dir)                   
         else:
             print("ERROR: images folder already exists")
     
         if not os.path.exists(masks_dir):
             os.makedirs(masks_dir)
         else:
             print("ERROR: masks folder already exists")
         
         original_label_locs = [file for file in os.listdir(root_dir_masks) if file.endswith(Ztag3 + '.tif')]   
         # Read whole segmentation label image
         whole_mask = imread(join(root_dir_masks, original_label_locs[0]))
     
         # Generate seperate instance mask from the segmentation labels
         generate_instance_mask(whole_mask, masks_dir)

         index = index + 1

def convert_channel3_png(root_dir_dataset, num_channels=3):
    """
    Convert original 1-channel grayscale images to 3-channel to feed into DCNN
    """
    for folder in os.listdir(root_dir_dataset):
        image_folder_dir = join(root_dir_dataset, folder, 'images')
        
        for images in os.listdir(image_folder_dir):
            image_filename = join(image_folder_dir, images)
  
            # When reading ome.xml file, set "is_ome = False" to disable reading metadata
            orig_img = imread(image_filename, is_ome = False) 
            concat_img = np.repeat(orig_img[..., np.newaxis], num_channels, -1)
            imsave(join(image_folder_dir, images), concat_img)
            print("saved file {}").format(images)

def change_file_name(root_dir_dataset):
    """Make the image filename the same as the case folder name
    """
    for case_folder in os.listdir(root_dir_dataset):
        images_folder_dir =  join(root_dir_dataset, case_folder, 'images')

        for images in os.listdir(images_folder_dir):
            # extension = splitext(images)[1]  # .ome.tif
            
            old_name = join(images_folder_dir, images)
            new_name = join(images_folder_dir, str(case_folder) + '.ome.tif')
            os.rename(old_name, new_name)       
            print("renamed file {}").format(new_name)
            
def convert_data_type(root_dir_dataset):
    """
    Convert 16-bit image to 8-bit image and save all as tiff format
    """
    image_ids = next(os.walk(root_dir_dataset))[1]
    
    for image_id in image_ids:
        data_dir = os.path.join(root_dir_dataset, image_id, "images")
        image_name = os.listdir(data_dir)[0]
        image_dir = os.path.join(data_dir, image_name)

        if image_name.endswith('.tif'):
            image = imread(image_dir)
            imsave(join(data_dir, image_name), image.astype(np.uint8))
         
        if image_name.endswith('.ome.tif'):
            image = imread(image_dir, is_ome = False)
            imsave(join(data_dir, image_name[:-8]+'.tif'), image.astype(np.uint8))
            os.remove(image_dir)
            #print(join(data_dir, image_name[:-8]+'.tif'))

def process_data(root_dir): 
    folder_list =  next(os.walk(root_dir))[1]
    for folder_name in folder_list:
        file_list = os.listdir(join(root_dir, folder_name))
        masks_dir = join(root_dir, folder_name,'masks')
        for file in file_list:
            if file.endswith('.png'):
                whole_mask = skimage.io.imread(join(root_dir, folder_name,file))
                generate_instance_mask(whole_mask, masks_dir)

            
# Define root_directory for segmentation labels and original images

root_dir_masks = '/path/to/label/'
root_dir_images = '/path/to/image'
dir_to_save = '/path/to/save'

# Executables
directory_organizer(root_dir_images, root_dir_masks, dir_to_save)
