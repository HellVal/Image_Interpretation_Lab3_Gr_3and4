# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:53:01 2021

@author: Pascal
"""

import torch.utils.data
import torch
import numpy as np
import h5py
import os.path
from sklearn.model_selection import train_test_split

class preprocessing(torch.utils.data.Dataset):
    def __init__(self, path, time_downsample_factor=1, num_channel=4):

        self.num_channel = num_channel
        self.time_downsample_factor = time_downsample_factor
        self.eval_mode = False
        # Open the data file
        self.data = h5py.File(path, "r", libver='latest', swmr=True)

        data_shape = self.data["data"].shape
        target_shape = self.data["gt"].shape
        self.num_samples = data_shape[0]

        if len(target_shape) == 3:
            self.eval_mode=True
            self.num_pixels = target_shape[0]*target_shape[1]*target_shape[2]
        else:
            self.num_pixels = target_shape[0]

        label_idxs = np.unique(self.data["gt"])
        self.n_classes = len(label_idxs)
        self.temporal_length = data_shape[-2]//time_downsample_factor

        print('Number of pixels: ', self.num_pixels)
        print('Number of classes: ', self.n_classes)
        print('Temporal length: ', self.temporal_length)
        print('Number of channels: ', self.num_channel)

    def return_labels(self):
        return self.data["gt"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        X = self.data["data"][idx]
        target = self.data["gt"][idx]

        # Convert numpy array to torch tensor
        X = torch.from_numpy(X)
        target = torch.from_numpy(np.array(target)).float()

        # if self.eval_mode:
        #     X = X.view()
        #     target = target.view()

        # Temporal down-sampling
        X = X[...,0::self.time_downsample_factor, :self.num_channel]

        # keep values between 0-1
        X = X * 1e-4

        #return X.float(), target.long()
        return X.float()
    
    
import matplotlib.pyplot as plt

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
# for i in range(len(labels_name)):
#     good_labels.append(labels_all[label_names_all.index(labels_name[i])])
    

def plot_bands(X):
    x = np.arange(X.shape[0])
    for i, band in enumerate(plotbands):
        plt.plot(x, X[:,i])

    plt.savefig("bands.png", dpi=300, format="png", bbox_inches='tight')
    

def h5builder(path,layer_name,layer_data):
    
    if os.path.isfile(path):
        f = h5py.File(path, "a")
        f.create_dataset(layer_name, data = layer_data)
        f.close()
    else:
        f = h5py.File(path, "w")
        f.create_dataset(layer_name, data = layer_data)
        f.close() 
    print('h5 layer successfull')


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
    


def calculate_ndvi(red, nir):


    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    ndvi = (nir.astype(float) - red.astype(float)) / (nir + red)
    return ndvi

#for regions with high content of atmospheric aerosol (e.g. rain, fog, dust, smoke, air pollution)
def calculate_arvi(nir, red, blue):

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    # Calculate NDVI
    arvi = (nir-(2*red)+blue)/(nir+(2*red)+blue)

    return arvi
    

#for monitoring the impact of seasonality, environmental stresses, applied pesticides on plant health.
def calculate_gci(nir, green):

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    # Calculate NDVI
    gci = nir/green - 1
    
    return gci
    
if __name__ == "__main__":

   # where the data is stored 
    data_path = "imgint_trainset.hdf5"
    
    #load the data
    traindataset = preprocessing(data_path)
    #get the labels
    gt_list_all = traindataset.return_labels()
    
    # get an overview over the used labels, filter out the used labels and pixels --> adapt it can be implemented static
    # label_list, numbers = overview(traindataset,gt_list_all)
    label_list = [36, 46, 42, 48, 30, 45, 50, 49, 38, 27, 20, 51, 21]
    # sorted list --> how many time the corp appears in the dataset
    numbers = [  47277,   61761,   75384,   82334,  127164,  180346,
            221464,  240629,  314101,  595184,  723058,  761008,
           3574346]
    
    #%%
    # set the numbers of the selected pixel in the dataset
    n_samples = 10
    n_crops = 13
    
    indexes = np.zeros((n_crops,n_samples))
    
    # to select the first n_samples in the code 
    for i in range(13):
        temp = np.where(np.array(label_list)[i] == np.array(gt_list_all))
        indexes[i,:] = temp[0][:n_samples]
        
    #%%
    
    duration = 71
    n_channels = 4
    output = np.zeros((n_crops,n_samples,duration,n_channels))
    #insert RGB and NIR to the ouput matrix 
    for i in range(13):
        output[i,:,:,:] = traindataset[indexes[i,:]]
        
    # select which is the step to select the pixels for validation  
    n_pix_val = 3
    
    validation = output[:,::n_pix_val,:,:]
    #mask to skip the validation data from the training data
    mask = np.mod(np.arange(n_samples),n_pix_val)!=0
    training  = output[:,mask,:,:]
    

    h5builder('training.hdf5', 'DATA', training)
    h5builder('training.hdf5', 'GT', labels)
    
    h5builder('validation.hdf5', 'DATA', validation)
    h5builder('validation.hdf5', 'GT', labels)
    
    
    
    

    
    
    
    
