#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:05:49 2021

@author: valerie.hellmueller
"""


# Data analyse

def show_nan(picture):
    
    i_n = np.argwhere(np.isnan(picture)
                        
    return i_n


nan_index = show_nan(xxx)
number_of_nan = np.size(nan_index)


a = np.eye(data_shape)
a[a[nan_index]]=0

#plot
imshow(a)
plt.show()



def show_zero(picture):
    
    i_z = np.where(picture == 0)[0]
    
    return i_z


zeros_index = show_zero(XXX)
number_of_zeros = np.size(zeros_index)

# plot

b = np.eye(data_shape)
b[b[nan_index]]=0


imshow(b)
plt.show()