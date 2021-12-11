# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 19:08:17 2021

@author: paimhof
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
    
    
    data_path = "imgint_testset_v2.hdf5"
    
    # dataloader
    dataset = dp.preprocessing(data_path)
    gt_list_all = dataset.return_labels()
    
    #%% reduce dimension in the data
    
    data = dataset[:].detach().cpu().numpy()
    
    data_adusted = np.transpose(data,(3,4,0,1,2))
    
    
    output = []
    
    for i_slice in data_adusted:
        _temp1 = []
        for j_slice in i_slice:
            _temp2 = j_slice.flatten()
            _temp1.append(_temp2)
        output.append(_temp1)
            
    data_adjusted_2 = np.transpose(np.array(output), (2,0,1))        
    
    #%% reduce dimension in the groundtruth
    gt = gt_list_all[:].flatten()
    
    
#%% select all labels which are not 0
    index = np.nonzero(gt)
    data_adjusted_3 = data_adjusted_2[index[0],:,:]
    
    test_gt = gt[index[0]]
    
    #%% compute features
    
    ndvi = hf.calculate_ndvi(data_adjusted_3[:,:,0], data_adjusted_3[:,:,3])
    arvi = hf.calculate_arvi(data_adjusted_3[:,:,3], data_adjusted_3[:,:,0],data_adjusted_3[:,:,2])
    gci = hf.calculate_gci(data_adjusted_3[:,:,3], data_adjusted_3[:,:,1])
    
    
    #%% stack to gether everything 
    
    red = data_adjusted_3[:,:,0]
    green = data_adjusted_3[:,:,1]
    blue = data_adjusted_3[:,:,2]
    nir = data_adjusted_3[:,:,3]
    
    test_data =  np.stack((red, green, blue, nir,ndvi,arvi,gci),axis = 2)
    
    
    hf.h5builder('test_data_final.hdf5', 'DATA', test_data)
    hf.h5builder('test_data_final.hdf5', 'GT', test_gt)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

        
        
        
        