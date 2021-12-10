# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:49:15 2021

@author: Pascal
"""

import torch
import numpy as np
import random

import datalaoder_preprocessing as dp
import helper_functions as hf


label_list = [36, 46, 42, 48, 30, 45, 50, 49, 38, 27, 20, 51, 21]
numbers = [  47277,   61761,   75384,   82334,  127164,  180346,
            221464,  240629,  314101,  595184,  723058,  761008,
           3574346]



if __name__ == "__main__":
    data_path = "imgint_trainset.hdf5"
    
    # dataloader
    dataset = dp.preprocessing(data_path)
    gt_list_all = dataset.return_labels()
#%%
    
    # mask for random selected samples 
    mask = np.zeros(np.shape(gt_list_all))
    
    #
    n_samples = 10000
    n_crops = len(label_list)
    
    
    for i in range(n_crops):
        #get all pixels of the acutal crop
        temp = np.where(np.array(label_list)[i] == np.array(gt_list_all))
        # take random samples of the crop
        random_samples = np.array(hf.quicksort(random.sample(temp[0].tolist(),n_samples)))

        # set random_selected values to 1
        mask[random_samples] = 1
    
    print("Finished indexing")
    
    mask = mask.astype('int32')
    #position of the samples
    idx = np.where(mask == 1)
    
    
    
  #%%  
    # load data
    data = dataset[idx].detach().cpu().numpy()
    #load groundtruth
    gt = gt_list_all[idx]
    
    red = data[:,:,0]
    green = data[:,:,1]
    blue = data[:,:,2]
    nir = data[:,:,3]
    
    # compute features 
    ndvi = hf.calculate_ndvi(data[:,:,0], data[:,:,3])
    arvi = hf.calculate_arvi(data[:,:,3], data[:,:,0],data[:,:,2])
    gci = hf.calculate_gci(data[:,:,3], data[:,:,1])
    
    # stack final all togheter data and features
    feature_data = np.stack((red, green, blue, nir,ndvi,arvi,gci),axis = 2)
    
    
    
    # create training and validation dataset
    # select which is the step to select the pixels for validation  
    n_pix_val = 4
    #select each n pixel for validation 
    val_data = feature_data[::n_pix_val,:,:]
    val_gt = gt[::n_pix_val]
    #mask to skip the validation data from the training data
    training_mask = np.mod(np.arange(n_samples*n_crops),n_pix_val)!=0
    train_data  = feature_data[training_mask,:,:]
    train_gt = gt[training_mask]
    
    
#%% 
    # #write data to hdf5 file
    # hf.h5builder('training_40000.hdf5', 'DATA', train_data)
    # hf.h5builder('training_40000.hdf5', 'GT', train_gt)
    
    # hf.h5builder('validation_40000_test.hdf5', 'DATA', val_data)
    # hf.h5builder('validation_40000_test.hdf5', 'GT', val_gt)


    
 
    

    
    
    
    
    
    
    
    