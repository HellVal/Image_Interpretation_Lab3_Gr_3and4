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

labels_name = ['Meadow','Winter wheat','Maize','Pasture','Sugar beet','Winter barley',
                    'Winter rapeseed','Vegetables','Potatoes','Wheat','Sunflowers','Vines','Spelt']
zip_iterator = zip(label_list, labels_name)

used_labels = dict(zip_iterator)

if __name__ == "__main__":
    
    
    data_path = "D:/ImageInterpretation/lab3/imgint_trainset.hdf5"
    
    # dataloader
    dataset = dp.preprocessing(data_path)
    gt_list_all = dataset.return_labels()
#%%
    
    # mask for random selected samples 
    mask = np.zeros(np.shape(gt_list_all))
    
    # number of randomly sampled pixels
    n_samples = 2000
    n_crops = len(label_list)
    
    
    for i in range(n_crops):
        print(i)
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
    
#%% initialize the final output matrix 
    output = np.zeros((n_crops*n_samples,71,7))
    gt_output   = np.zeros((n_crops*n_samples))
    
    stepsize = 1000
    
    for i in range(0,(n_crops*n_samples),stepsize):
        print(i)
        temp_idx = idx[0][i:i+stepsize]
        temp_data = dataset[temp_idx].detach().cpu().numpy()
        temp_gt = gt_list_all[temp_idx]
        print('DATA LOADED: ' +str(i))
        
        #compute the feauters of the loaded data
        ndvi = hf.calculate_ndvi(temp_data[:,:,0], temp_data[:,:,3])
        arvi = hf.calculate_arvi(temp_data[:,:,3], temp_data[:,:,0],temp_data[:,:,2])
        gci = hf.calculate_gci(temp_data[:,:,3], temp_data[:,:,1])    
        
        
        red = temp_data[:,:,0]
        green = temp_data[:,:,1]
        blue = temp_data[:,:,2]
        nir = temp_data[:,:,3]
        
        output[i:(i+stepsize),:,:] = np.stack((red, green, blue, nir,ndvi,arvi,gci),axis = 2)
        gt_output[i:(i+stepsize)] = temp_gt
        
    feature_data = output
    gt = gt_output
        
            
    
#%%    
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
    #write data to hdf5 file
    hf.h5builder('training_2000.hdf5', 'DATA', train_data)
    hf.h5builder('training_2000.hdf5', 'GT', train_gt)
    
    hf.h5builder('validation_2000.hdf5', 'DATA', val_data)
    hf.h5builder('validation_2000.hdf5', 'GT', val_gt)