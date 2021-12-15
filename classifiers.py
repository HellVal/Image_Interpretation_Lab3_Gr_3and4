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

rootVal = r"P:/pf/pfstud/II_Group3/Lab 3/validation_40000.hdf5"
hv = h5py.File(rootVal, 'r')

rootTest = r"P:/pf/pfstud/II_Group3/Lab 3/test_data_final.hdf5"
htest = h5py.File(rootTest, 'r')
featuresUsed = [0,1,2,3,4,5,6]

#%%
# reading the data
data = ht['DATA']
gt = ht['GT']
#classes = [0,1,2,3,4,5,6,7,8,9,10,11,12]

# number of pixels and classes
nPixels = data.shape[0]
#nClasses = gt.shape[0]
nDays = data.shape[1]-1
nFeatures = len(featuresUsed)


#%%
xTrainTemp = np.empty((0,nDays * nFeatures),dtype = 'float64')

#Building the feature vector
temp = np.empty((nPixels,0))
for i in featuresUsed:
    temp = np.append(temp, data[:,0:12,i], axis = 1)
    temp = np.append(temp, data[:,13:,i], axis = 1)

xTrainTemp = np.append(xTrainTemp, temp, axis = 0)

#%%
indices = np.where(np.all(~np.isnan(xTrainTemp), axis = 1))
xTrain = xTrainTemp[indices[0],:]
labels_train = gt[indices[0]]

#%% Validation data
dataVal = hv['DATA']
gtVal = hv['GT']

# number of pixels and classes
nPixelsVal = dataVal.shape[0]
#nClassesVal = gtVal.shape[0]
nDaysVal = dataVal.shape[1]-1
nFeatures = len(featuresUsed)

#%%
xValTemp = np.empty((0,nDaysVal * nFeatures),dtype = 'float64')


#Building the feature vector

temp = np.empty((nPixelsVal,0))
for i in featuresUsed:
    temp = np.append(temp, dataVal[:,0:12,i], axis = 1)
    temp = np.append(temp, dataVal[:,13:,i], axis = 1)

xValTemp = np.append(xValTemp, temp, axis = 0)

#%%
indicesVal = np.where(np.all(~np.isnan(xValTemp), axis = 1))
xVal = xValTemp[indicesVal[0],:]
labels_Val = gtVal[indicesVal[0]]

#%% Test data
dataTest = htest['DATA']
labelsTestTemp = htest['GT']

# number of pixels and classes
nPixelsTest = dataTest.shape[0]
nDaysTest = dataTest.shape[1]-1
nFeatures = len(featuresUsed)

#%%
xTestTemp = np.empty((0,nDaysTest * nFeatures),dtype = 'float64')


#Building the feature vector
temp = np.empty((nPixelsTest,0))
for i in featuresUsed:
    temp = np.append(temp, dataTest[:,0:12,i], axis = 1)
    temp = np.append(temp, dataTest[:,13:,i], axis = 1)

xTestTemp = np.append(xTestTemp, temp, axis = 0)


#%%
indicesTest = np.where(np.all(~np.isnan(xTestTemp), axis = 1))
xTest = xTestTemp[indicesTest[0],:]
labelsTest = labelsTestTemp[indicesTest[0]].astype(int)
#labelsTest = np.reshape(labelsTest,(labelsTest.shape[0],)).astype(int)
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
    RandomForestClassifier(max_depth=300, n_estimators=100, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# #Training the classifier
# for name, clf in zip(names, classifiers):
#         ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#         clf.fit(xTrain, lasbels_train)


#%%
classfierNumber = 0
classifier = KNeighborsClassifier(3)
classifier = classifier.fit(xTrain, labels_train)

#%% Classification of Validation data
classifierOut = classifier.predict(xVal)

#%% Evaluation
labelsVal = [20,21,27,30,36,38,42, 45,46,48,49,50,51]
cm = confusion_matrix(labels_Val, classifierOut, labels=labelsVal)

# Number of annotated examples per class (TP + FN)
labelPerClass = np.sum(cm, axis=1)

# True positives per class (TP)
truePositivesSvm = np.diag(cm)

# Number of total predictions per class (TP + FP)
predPerClassSvm = np.sum(cm, axis=0)

# Overall accuracy (one value per classifier)
overallAccuracySvm = np.sum(truePositivesSvm) / np.sum(labelPerClass)

# Producer's accuracy (one value per class per classifier)
Recall = truePositivesSvm / labelPerClass

# User's accuracy (one value per class per classifier)
Precision = truePositivesSvm / predPerClassSvm

# F1
F1 = 2 * (Precision * Recall)/(Precision + Recall)

#%% Showing the results

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelsVal)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)


output_file_name = names[classfierNumber] + '.txt'

if os.path.exists(output_file_name):
    os.remove(output_file_name)

original_stdout = sys.stdout
with open(output_file_name, 'w') as f:
    sys.stdout = f

    print(names[classfierNumber],'\n')
    print(f'Overall accuracy: {overallAccuracySvm}')
    print(f'Recall: {Recall}')
    print(f'Precision: {Precision}')
    print(f'F1: {F1}')
    print(f'Confusion matrix:')
    pprint(cm)

    sys.stdout = original_stdout

print(f'Outputs saved to {output_file_name}')

#%% Classification of test data
xTest1 = xTest[0:1000000]
labelsTest1 = labelsTest[0:1000000]
classifierOut = classifier.predict(xTest1)

#%% Evaluation
cm = confusion_matrix(labelsTest1, classifierOut,labels=labelsVal)

# Number of annotated examples per class (TP + FN)
labelPerClass = np.sum(cm, axis=1)

# True positives per class (TP)
truePositivesSvm = np.diag(cm)

# Number of total predictions per class (TP + FP)
predPerClassSvm = np.sum(cm, axis=0)

# Overall accuracy (one value per classifier)
overallAccuracySvm = np.sum(truePositivesSvm) / np.sum(labelPerClass)

# Producer's accuracy (one value per class per classifier)
Recall = truePositivesSvm / labelPerClass

# User's accuracy (one value per class per classifier)
Precision = truePositivesSvm / predPerClassSvm

# F1
F1 = 2 * (Precision * Recall)/(Precision + Recall)


#%% Showing the results

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelsVal)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)


output_file_name = names[classfierNumber] + '_test.txt'

if os.path.exists(output_file_name):
    os.remove(output_file_name)

original_stdout = sys.stdout
with open(output_file_name, 'w') as f:
    sys.stdout = f

    print(names[classfierNumber],'\n')
    print(f'Overall accuracy: {overallAccuracySvm}')
    print(f'Recall: {Recall}')
    print(f'Precision: {Precision}')
    print(f'F1: {F1}')
    print(f'Confusion matrix:')
    pprint(cm)

    sys.stdout = original_stdout

print(f'Outputs saved to {output_file_name}')