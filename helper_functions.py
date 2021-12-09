# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 09:15:45 2021

@author: Pascal
"""

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import h5py
import torch

#%% usefull functions

def quicksort(array):
    # If the input array contains fewer than two elements,
    # then return it as the result of the function
    if len(array) < 2:
        return array

    low, same, high = [], [], []

    # Select your `pivot` element randomly
    pivot = array[random.randint(0, len(array) - 1)]

    for item in array:
        # Elements that are smaller than the `pivot` go to
        # the `low` list. Elements that are larger than
        # `pivot` go to the `high` list. Elements that are
        # equal to `pivot` go to the `same` list.
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)

    # The final result combines the sorted `low` list
    # with the `same` list and the sorted `high` list
    return quicksort(low) + same + quicksort(high)



def h5builder(path,layer_name,layer_data):
    
    if os.path.isfile(path):
        f = h5py.File(path, "a")
        f.create_dataset(layer_name, data = layer_data)
        f.close()
    else:
        f = h5py.File(path, "w")
        f.create_dataset(layer_name, data = layer_data)
        f.close() 
    print('h5 layer successfull')


def test_data_reshaper(data, gt):
    
    #number of patches
    n_patches = 8357
    n_channels = 4
    n_days = 71
    s_patch = 24
                
    data_res = torch.zeros((n_patches*s_patch*s_patch,n_days,n_channels))
    gt_res = np.zeros((n_patches*s_patch*s_patch,1))
    
    for i in range(n_patches):
        data_res[i*(s_patch*s_patch):(i+1)*(s_patch*s_patch),:,:] = torch.reshape(data[i],(s_patch*s_patch,n_days,n_channels))
        gt_res[i*(s_patch*s_patch):(i+1)*(s_patch*s_patch),0] = np.reshape(gt[i],(s_patch*s_patch))
        
    #data_res = data_res.detach().cpu().numpy()
    gt_res.astype(np.int64)
        
    return (data_res, gt_res)


#%% feature functions
def calculate_ndvi(red, nir):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    # Calculate NDVI
    ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
    return ndvi

#for regions with high content of atmospheric aerosol (e.g. rain, fog, dust, smoke, air pollution)
def calculate_arvi(nir, red, blue):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')  
    # Calculate NDVI
    arvi = (nir-(2*red)+blue)/(nir+(2*red)+blue)
    return arvi
    

#for monitoring the impact of seasonality, environmental stresses, applied pesticides on plant health.
def calculate_gci(nir, green):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    # Calculate NDVI
    gci = nir/green - 1
    return gci


# computes the mean of a channel
def mean_calculator(channel):
    mean_value = np.mean(channel)
    return (mean_value)

def median_finder(channel):
    median_value = np.median(channel)
    return(median_value)


# finds the min value of a channel 
def min_finder(channel):
    min_value = np.min(channel)
    return(min_value)

# finds the max value of a channel 
def max_finder(channel):
    max_value = np.max(channel)
    return max_value
    
    
