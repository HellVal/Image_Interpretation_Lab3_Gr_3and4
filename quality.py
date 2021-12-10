#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:05:49 2021

@author: valerie.hellmueller
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


# Input Parameters

root = r'/Users/valerie.hellmueller/Documents/GitHub/Image_Interpretation_Lab3_Gr_3and4' #change path
files = ["/training_40000_random.hdf5"]


# Loading Data
file = root + files[0]
h5 = h5py.File(file, 'r')
(h5.keys())

img_names = ['DATA']
#data_shape = len(img)


for j in range(1): #img_names.size    len(img_names)
    img = h5[img_names[j]]
     

    rgb = np.zeros((img[1].shape[0],img[1].shape[1] ,4))
    mean = np.zeros((img[1].shape[0],img[1].shape[1] ,4))

for i in range(img.shape[0]):
        # Taking images out
        
     rgb = img[i]

# %% Data anlyse


def show_nan(picture):
    
    i_n = np.argwhere(np.isnan(picture))
    n_n = np.isnan(picture).sum()
                      
    return i_n, n_n
                      


nan_index, number_of_nan = show_nan(rgb)

#nan_index = int(nan_index == True)





#a = np.eye(data_shape,data_shape)
#a[a[nan_index]]=0

#plot

#plt.imshow(a)

#%%

def show_zero(picture):
    
    i_z = np.argwhere(picture == 0)
    n_z = np.sum(picture == 0)
    
    return i_z, n_z


zeros_index, number_of_zeros = show_zero(rgb)
#zeros_index = int(zeros_index == True)

# plot

#b = np.eye(data_shape,data_shape)
#b[b[nan_index]]=0


#plt.imshow(b)
