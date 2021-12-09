# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:53:01 2021

@author: Pascal
"""

import torch.utils.data
import torch
import numpy as np
import matplotlib.pyplot as plt
import random 
import helper_functions as hf
import datalaoder_preprocessing as dp


    
colordict = {'B04': '#a6cee3', 'NDWI': '#1f78b4', 'NDVI': '#b2df8a', 'RATIOVVVH': '#33a02c', 'B09': '#fb9a99',
             'B8A': '#e31a1c', 'IRECI': '#fdbf6f', 'B07': '#ff7f00', 'B12': '#cab2d6', 'B02': '#6a3d9a', 'B03': '#0f1b5f',
             'B01': '#b15928', 'B10': '#005293', 'VH': '#98c6ea', 'B08': '#e37222', 'VV': '#a2ad00', 'B05': '#69085a',
             'B11': '#007c30', 'NDVVVH': '#00778a', 'BRIGHTNESS': '#000000', 'B06': '#0f1b5f'}
plotbands = ["B02", "B03", "B04", "B08"]

labels_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]

label_names_all = ['Unknown', 'Apples', 'Beets', 'Berries', 'Biodiversity area', 'Buckwheat',
               'Chestnut', 'Chicory', 'Einkorn wheat', 'Fallow', 'Field bean', 'Forest',
               'Gardens', 'Grain', 'Hedge', 'Hemp', 'Hops', 'Legumes', 'Linen', 'Lupine',
               'Maize', 'Meadow', 'Mixed crop', 'Multiple', 'Mustard', 'Oat', 'Pasture', 'Pears',
               'Peas', 'Potatoes', 'Pumpkin', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Stone fruit',
               'Sugar beet', 'Summer barley', 'Summer rapeseed', 'Summer wheat', 'Sunflowers',
               'Tobacco', 'Tree crop', 'Vegetables', 'Vines', 'Wheat', 'Winter barley',
               'Winter rapeseed', 'Winter wheat']

labels = [21, 51, 20, 27, 38, 49, 50, 45, 30, 48, 42, 46, 36]
labels_name = ['Meadow','Winter wheat','Maize','Pasture','Sugar beet','Winter barley',
                    'Winter rapeseed','Vegetables','Potatoes','Wheat','Sunflowers','Vines','Spelt']
zip_iterator = zip(labels, labels_name)

used_labels = dict(zip_iterator)


def overview(data,gt):
    
    labels_counts, pix_counts = np.unique(gt, return_counts=True)
    masked_counts = np.zeros(13)
    for i in range(13):
        temp = np.where(labels[i]==labels_counts)
        masked_counts[i] = pix_counts[temp[0][0]]

    inds = masked_counts.argsort()
    labels_sorted = np.array(labels)[inds]
    masked_counts_sorted  = masked_counts[inds]
    
    return (labels_sorted,masked_counts_sorted)

def plot_bands(X):
    x = np.arange(X.shape[0])
    for i, band in enumerate(plotbands):
        plt.plot(x, X[:,i])

    plt.savefig("bands.png", dpi=300, format="png", bbox_inches='tight')
    
    


    
if __name__ == "__main__":
    # training data = 0, test data = 1
    set_variable = 1
   
    # set path of the data
    if set_variable == 1:
        data_path = 'imgint_testset_v2.hdf5'
        # insert here reshape function for testdata
    else:
        data_path = "imgint_trainset.hdf5"
            
    #load data
    dataset = dp.preprocessing(data_path)
    #get the labels
    gt_list_all = dataset.return_labels()
    
    
    # reshape testdata
    if set_variable == 1:
        (dataset,gt_list_all) = hf.test_data_reshaper(dataset,gt_list_all)
    
    # get an overview over the used labels, filter out the used labels and pixels --> adapt it can be implemented static
    # label_list, numbers = overview(dataset,gt_list_all)
    label_list = [36, 46, 42, 48, 30, 45, 50, 49, 38, 27, 20, 51, 21]
    # sorted list --> how many time the corp appears in the dataset
    numbers = [  47277,   61761,   75384,   82334,  127164,  180346,
            221464,  240629,  314101,  595184,  723058,  761008,
           3574346]
    
    #%%
    # set the numbers of the selected pixel in the dataset
    n_samples = 1000
    n_crops = len(label_list)
    
    indexes = np.zeros((n_crops,n_samples))
            
    #random selection of the pixels
    for i in range(n_crops):
        #get all pixels of the acutal crop
        temp = np.where(np.array(label_list)[i] == np.array(gt_list_all))
        # select random n_sample out of this pixels and sort it
        indexes[i,:] = np.array(hf.quicksort(np.array(random.sample(temp[0].tolist(),n_samples))))
        print(used_labels[labels[i]]+' random pixel selection done!')
        
    #%%
    channels = ['R','G','B','NIR','NDVI','ARVI','GCI']
    duration = 71
    n_channels = len(channels)
    
    #genereate outputMatrix
    output = np.zeros((n_crops,n_samples,duration,n_channels))

    indexes = indexes.astype(np.int64)
    for i in range(n_crops):
        temp = dataset[indexes[i,:]]
        output[i,:,:,:4] = temp
        output[i,:,:,4] = hf.calculate_ndvi(temp[:,:,0].detach().cpu().numpy(), temp[:,:,3].detach().cpu().numpy())
        output[i,:,:,5] = hf.calculate_arvi(temp[:,:,3].detach().cpu().numpy(), temp[:,:,0].detach().cpu().numpy(),temp[:,:,2].detach().cpu().numpy())
        output[i,:,:,6] = hf.calculate_gci(temp[:,:,3].detach().cpu().numpy(), temp[:,:,1].detach().cpu().numpy())
        print(used_labels[labels[i]]+' feature computation done!')
    
    

    if set_variable == 1:
        hf.h5builder('test_data_set.hdf5', 'DATA', output)
        hf.h5builder('test_data_set.hdf5', 'GT', labels)
    else:
        # select which is the step to select the pixels for validation  
        n_pix_val = 4
        
        validation = output[:,::n_pix_val,:,:]
        #mask to skip the validation data from the training data
        mask = np.mod(np.arange(n_samples),n_pix_val)!=0
        training  = output[:,mask,:,:]
        
    
        # hf.h5builder('training_1000_test.hdf5', 'DATA', training)
        # hf.h5builder('training_1000_test.hdf5', 'GT', labels)
        
        # hf.h5builder('validation_1000_test.hdf5', 'DATA', validation)
        # hf.h5builder('validation_1000_test.hdf5', 'GT', labels)
        
    
    
    

    
    
    
    
