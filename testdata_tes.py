# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:19:43 2021

@author: Pascal
"""

import torch
import numpy as np

import datalaoder_preprocessing as dp
import helper_functions as hf



data_path = 'imgint_testset_v2.hdf5'

#load the data
dataset = dp.preprocessing(data_path)
gt_list_all = dataset.return_labels()
# reshape the data
(dataset,gt_list_all) = hf.test_data_reshaper(dataset,gt_list_all)

#%%


index = np.nonzero(gt_list_all)
dataset_filtered = dataset[index[0].tolist(),:,:]

#%% compute the featuers
dataset_filtered = dataset_filtered.detach().cpu().numpy()

ndvi = hf.calculate_ndvi(dataset_filtered[:,:,0], dataset_filtered[:,:,3])
arvi = hf.calculate_arvi(dataset_filtered[:,:,3], dataset_filtered[:,:,0],dataset_filtered[:,:,2])
gci = hf.calculate_gci(dataset_filtered[:,:,3], dataset_filtered[:,:,1])
#%%
output = np.zeros((926021,71,7))
output[:,:,:4] = dataset_filtered
output[:,:,4] = ndvi
output[:,:,5] = arvi
output[:,:,6] = gci

#%% write everything to h5 file


hf.h5builder('test_data.hdf5', 'DATA', output)
hf.h5builder('test_data.hdf5', 'GT', gt_list_all[index[0].tolist()])

