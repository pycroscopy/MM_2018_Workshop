# -*- coding: utf-8 -*-
"""
Utility functions for DL tutorial

Created on Mon Jul 30 20:57:13 2018

@author: Maxim Ziatdinov
"""

import h5py
import numpy as np
from scipy import ndimage
import cv2
from skimage.feature import blob_log

def resize_(input_, ref_):
    '''Upsampling with bilinear interpolation (not available directly from Keras)'''
    import tensorflow as tf
    W, H = ref_
    return tf.image.resize_bilinear(input_, [W, H])


def load_training_data_(hf_file):
    '''Load training images and corresponding ground truth data'''
    with h5py.File(hf_file, 'r') as f:
        return f['Images'][:], f['GT'][:]
        

def tf_format(image_data, image_size):
    '''Change image format to keras/tensorflow format'''
    
    image_data = image_data.reshape(image_data.shape[0], image_size[0], image_size[1], 1)
    image_data = image_data.astype('float32')
    image_data = (image_data - np.amin(image_data))/np.ptp(image_data)
    return image_data

def coord_edges(coordinates, target_size, dist_edge):
    '''Remove image edges'''
    
    return [coordinates[0] > target_size[0] - dist_edge, coordinates[0] < dist_edge,
            coordinates[1] > target_size[0] - dist_edge, coordinates[1] < dist_edge]

def find_com(image_data):
    '''Find atoms via center of mass methods'''
    
    labels, nlabels = ndimage.label(image_data)
    coordinates = np.array(ndimage.center_of_mass(image_data, labels, np.arange(nlabels)+1))
    coordinates = coordinates.reshape(coordinates.shape[0], 2)
    return coordinates

def rem_coord(coordinates, target_size, dist_edge):
    '''Remove coordinates at the image edges'''
    
    coord_to_rem = [idx for idx, c in enumerate(coordinates) if any(coord_edges(c, target_size, dist_edge))]
    coord_to_rem = np.array(coord_to_rem, dtype = int)
    coordinates = np.delete(coordinates, coord_to_rem, axis = 0)
    return coordinates

def get_all_coordinates(decoded_imgs, target_size, method = 'LoG',
                    min_sigma = 1.5, max_sigma = 10, threshold = 0.8, dist_edge = 3):
    '''Extract all atomic coordinates in image via LoG or CoM methods & store data as a dictionary (key: frame number)'''
   
    d_coord = {}
    for i, decoded_img in enumerate(decoded_imgs):
        coordinates = np.empty((0,2))
        category = np.empty((0,1))
        for ch in range(decoded_img.shape[2]-1):
            _, decoded_img_c = cv2.threshold(decoded_img[:,:,ch], threshold, 1, cv2.THRESH_BINARY)
            if method == 'LoG':
                coord = blob_log(decoded_img_c, min_sigma=min_sigma, max_sigma=max_sigma)
            elif method == 'CoM':
                coord = find_com(decoded_img_c)
            coord_ch = rem_coord(coord, target_size, dist_edge)
            category_ch = np.zeros((coord_ch.shape[0], 1))+ch  
            coordinates = np.append(coordinates, coord_ch, axis = 0)
            category = np.append(category, category_ch, axis = 0)
        d_coord[i] = np.concatenate((coordinates, category), axis = 1)
        
    print("Atomic/defect coordinates extracted.\n")
    
    return d_coord