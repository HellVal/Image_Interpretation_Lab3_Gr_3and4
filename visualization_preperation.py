# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 17:47:56 2021

@author: Pascal
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import datalaoder_preprocessing as dp
import helper_functions as hf
#%%

# open results 
f = h5py.File('./lab_3_yatao/result_stacking_pixels.hdf5', 'r')

# dataloader for gt
data_path ='imgint_testset_v2.hdf5'
dataset = dp.preprocessing(data_path)
gt_list_all = dataset.return_labels()
#%% get all labels and flatten the matrix to a vector
gt = gt_list_all[:].flatten()

#%% find all nonzero position in gt
index = np.nonzero(gt)

#%% 
output = np.zeros(gt.shape)
#%% write predicted labels to output vector
L = f['Label']
labels = L[:]

output[index[0]] = labels

#%% reshape output to correct shape

test = np.reshape(output,(8358,24,24))


hf.h5builder('./lab_3_yatao/vis_stacking_pixels.hdf5', 'DATA', test)

f.close()






