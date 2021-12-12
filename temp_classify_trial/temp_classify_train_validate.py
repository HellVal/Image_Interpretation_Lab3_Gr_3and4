# -*- coding: utf-8 -*-
"""
Created on Wed Dec 1 12:29:33 2021
@author: Yatao
@purpose: classification for time-series data
"""
import os
import h5py
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from warnings import filterwarnings
filterwarnings('ignore')


def logistic_classifier(_train_data_vector, _train_gt_vector, _valid_data_vector):
    _lr = LogisticRegression()
    _lr.fit(_train_data_vector, _train_gt_vector)
    return _lr.predict(_valid_data_vector)


def stacking_vector(_data, _gt):
    # reshape ground truth
    _gt_vector = []
    for i_num in range(len(_gt)):
        _one_gt = np.full(shape=_data.shape[1], fill_value=_gt[i_num], dtype=int)
        _gt_vector = np.concatenate([_gt_vector, _one_gt])
    _gt_vector = np.array(_gt_vector)
    # reshape data
    _data_vector = []
    for i_slice in _data:
        for j_slice in i_slice:
            _data_vector.append(j_slice.ravel())
    _data_vector = np.array(_data_vector)

    # exclude pixels that have nan value
    indices = np.array(np.where(np.all(~np.isnan(_data_vector), axis=1)))
    _data_vector_r = _data_vector[indices[0], :]
    _gt_vector_r = _gt_vector[indices[0]]
    return _data_vector_r, _gt_vector_r


def time_series_stacking_classify(_train_data, _train_gt, _valid_data, _valid_gt):
    print("\nAlgorithm: time-series classification by stacking per-pixel features into a vector...")

    # data size: class_number * pixel_number * temporal_length * feature_length
    # ==> (class_number * pixel_number) * (temporal_length*feature_length)
    _train_data_vector, _train_gt_vector = stacking_vector(_train_data, _train_gt)
    _valid_data_vector, _valid_gt_vector = stacking_vector(_valid_data, _valid_gt)
    print("Train data size: {}, {}, {}, {}".format(_train_data.shape, _train_gt.shape,
                                                   _train_data_vector.shape, _train_gt_vector.shape))
    print("Valid data size: {}, {}, {}, {}".format(_valid_data.shape, _valid_gt.shape,
                                                   _valid_data_vector.shape, _valid_gt_vector.shape))

    # classification
    classifier_name = ['Logistics regression']
    classifier_func = [logistic_classifier]
    for i in range(len(classifier_name)):
        print("\n{}:".format(classifier_name[i]))
        _begin_time = time.time()
        _predict_valid_label = classifier_func[i](_train_data_vector, _train_gt_vector, _valid_data_vector)
        print("accuracy: {}\nprecision: {}\nrecall: {}\nf1 Score: {}".format(
            accuracy_score(_predict_valid_label, _valid_gt_vector),
            precision_score(_predict_valid_label, _valid_gt_vector, average=None),
            recall_score(_predict_valid_label, _valid_gt_vector, average=None),
            f1_score(_predict_valid_label, _valid_gt_vector, average=None)))
        print('memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print('time: {} s'.format(time.time() - _begin_time))


if __name__ == '__main__':
    # root = r"C:\ETH Course\Image interpretation\time_series\lab_3\\"
    root = "/home/yatao/time_series/lab_3"
    if os.path.exists(root):
        print("path exists...")
    # <KeysViewHDF5 ['DATA', 'GT']>
    train_h5 = h5py.File(root + '/training_40000.hdf5', 'r')
    train_data, train_GT = np.array(train_h5['DATA']), np.array(train_h5['GT'])
    # print("Train data size: {}, {}".format(train_data.shape, train_GT.shape))
    valid_h5 = h5py.File(root + '/validation_40000.hdf5', 'r')
    valid_data, valid_GT = np.array(valid_h5['DATA']), np.array(valid_h5['GT'])
    # print("Valid data size: {}, {}".format(valid_data.shape, valid_GT.shape))
    print(np.array(train_GT))

    print('before inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    begin_time = time.time()

    # data size: class_number * pixel_number * temporal_length * feature_length
    time_series_stacking_classify(train_data, train_GT, valid_data, valid_GT)
    print('stacking inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print('stacking inference time: {}'.format(time.time() - begin_time))
