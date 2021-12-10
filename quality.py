#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:05:49 2021

@author: valerie.hellmueller
"""

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image




# Data analyse

def show_nan(picture):
    
    i_n = np.argwhere(np.isnan(picture))
    n_n = np.isnan(picture).sum()
                      
    return i_n, n_n
                      


nan_index, number_of_nan = show_nan(img)

#nan_index = int(nan_index == True)





#a = np.eye(data_shape,data_shape)
#a[a[nan_index]]=0

#plot

#plt.imshow(a)



def show_zero(picture):
    
    i_z = np.argwhere(picture == 0)
    n_z = np.sum(picture == 0)
    
    return i_z, n_z


zeros_index, number_of_zeros = show_zero(img)
#zeros_index = int(zeros_index == True)

# plot

#b = np.eye(data_shape,data_shape)
#b[b[nan_index]]=0


#plt.imshow(b)
