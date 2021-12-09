# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 09:22:10 2021

@author: Pascal
"""

import torch.utils.data
import torch
import numpy as np
import h5py



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