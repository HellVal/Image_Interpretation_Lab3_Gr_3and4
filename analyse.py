#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:05:49 2021

@author: valerie.hellmueller
"""


# Data analyse

def show_nan(picture):
    
    index = np.argwhere(np.isnan(picture)
                        
    return index


nan_index = show_nan(data)
number_of_nan = np.size(nan_index)





def show_zero(picture):
    
    index = np.where(picture == 0)[0]
    
    
    return index


zeros_index = show_zero(data)
number_of_zeros = np.size(zeros_index)