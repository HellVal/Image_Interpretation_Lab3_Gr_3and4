# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:22:23 2021

@author: jfandre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:18:09 2021

@author: jfandre
"""

#Loading packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Confusion matrix
from sklearn.metrics import cohen_kappa_score

import sys
from pprint import pprint

#Loading data
root = r"P:/pf/pfstud/II_Group3/Lab 3/training_40000.hdf5"
ht = h5py.File(root, 'r')

rootVal = r"H:/Desktop/Image_Interpretation_Lab3_Gr_3and4-yatao/Image_Interpretation_Lab3_Gr_3and4-yatao/visualize_result/result_stacking_pixels.hdf5"
hv = h5py.File(rootVal, 'r')

rootTest = r"P:/pf/pfstud/II_Group3/Lab 3/test_data_final.hdf5"
htest = h5py.File(rootTest, 'r')

#%% reading the data
gtTest = htest['GT']
predictionTest = hv['Label']

#%%
labelsVal = [20,21,27,30,36,38,42, 45,46,48,49,50,51]
cm = confusion_matrix(gtTest, predictionTest, labels=labelsVal)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelsVal)
fig, ax = plt.subplots(figsize=(13,13))
disp.plot(ax=ax)