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
root = r"H:/Desktop/II lab3/training_10000.hdf5"
hf = h5py.File(root, 'r')

#%%
# reading the data
data = hf['DATA']
gt = hf['GT']

# number of pixels and classes
nPixels = data.shape[1]
nClasses = gt.shape[0]
nDays = data.shape[2]
nFeatures = data.shape[3]

x_train1 = np.stack(data, axis = 0)

featuresUsed = [0,1,2,3,4,5,6]
classes = [0,1,2,3,4,5,6,7,8,9,10,11,12]
#%%
xTrain = np.empty((0,nDays * nFeatures),dtype = 'float64')


#Building the feature vector
for j in classes:
    temp = np.empty((nPixels,0))
    for i in featuresUsed:
        temp = np.append(temp, data[j,:,:,i], axis = 1)
    
    xTrain = np.append(xTrain, temp, axis = 0)
    

labels_train = np.repeat(gt, nPixels)

#%%
#Initialising classifier
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifier = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

#Training the classifier
for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(xTrain, lasbels_train)

#%%
indices = np.where(np.all(~np.isnan(xTrain), axis = 1))
xTrain2 = xTrain[indices[0],:]
labels_train2 = labels_train[indices[0]]
#%%
classifier = KNeighborsClassifier(3)
classifier.fit(xTrain2, labels_train2)

#%% Validation data
rootVal = r"H:/Desktop/II lab3/validation_10000.hdf5"

#Evaluation