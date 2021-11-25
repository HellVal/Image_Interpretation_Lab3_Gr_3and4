#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:48:51 2021

@author: valerie.hellmueller

https://eos.com/blog/6-spectral-indexes-on-top-of-ndvi-to-make-your-vegetation-analysis-complete/
"""
# import rasterio
import numpy as np

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
