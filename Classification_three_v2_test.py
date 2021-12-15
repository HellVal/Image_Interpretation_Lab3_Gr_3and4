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


def rf_classifier(_train_data_vector, _train_gt_vector, _test_data_vector):
    _rf = RandomForestClassifier(n_estimators=100)
    _rf.fit(_train_data_vector, _train_gt_vector)
    return _rf.predict(_test_data_vector)


def svm_classifier(_train_data_vector, _train_gt_vector, _test_data_vector):
    _svm = LinearSVC()
    _svm.fit(_train_data_vector, _train_gt_vector)
    return _svm.predict(_test_data_vector)


def logistic_classifier(_train_data_vector, _train_gt_vector, _test_data_vector):
    _lr = LogisticRegression()
    _lr.fit(_train_data_vector, _train_gt_vector)
    return _lr.predict(_test_data_vector)


def knn_classifier(_train_data_vector, _train_gt_vector, _test_data_vector):
    _knn = KNeighborsClassifier(n_neighbors=3)
    _knn.fit(_train_data_vector, _train_gt_vector)
    return _knn.predict(_test_data_vector)


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


def stacking_vector_test(_data, _gt):
    # reshape ground truth
    _gt_vector = np.array(_gt).astype(int)
    # reshape data
    _data_vector = []
    for j_slice in _data:
        _data_vector.append(j_slice.ravel())
    _data_vector = np.array(_data_vector)

    # exclude pixels that have nan value
    indices = np.array(np.where(np.all(~np.isnan(_data_vector), axis=1)))
    _data_vector_r = _data_vector[indices[0], :]
    _gt_vector_r = _gt_vector[indices[0]]
    return _data_vector_r, _gt_vector_r


def time_series_stacking_classify(_train_data, _train_gt, _test_data, _test_gt):
    print("\nAlgorithm: time-series classification by stacking per-pixel features into a vector...")

    # data size: class_number * pixel_number * temporal_length * feature_length
    # ==> (class_number * pixel_number) * (temporal_length*feature_length)
    _train_data_vector, _train_gt_vector = stacking_vector(_train_data, _train_gt)
    _test_data_vector, _test_gt_vector = stacking_vector_test(_test_data, _test_gt)
    print("Train data size: {}, {}, {}, {}".format(_train_data.shape, _train_gt.shape,
                                                   _train_data_vector.shape, _train_gt_vector.shape))
    print("test data size: {}, {}, {}, {}".format(_test_data.shape, _test_gt.shape,
                                                  _test_data_vector.shape, _test_gt_vector.shape))

    # classification
    classifier_name = ['Random forest', 'Support vector machine', 'Logistics regression', 'KNN neighborhood']
    classifier_func = [rf_classifier, svm_classifier, logistic_classifier, knn_classifier]
    for i in range(len(classifier_name)):
        print("\n{}:".format(classifier_name[i]))
        _begin_time = time.time()
        _predict_test_label = classifier_func[i](_train_data_vector, _train_gt_vector, _test_data_vector)
        print("accuracy: {}\nprecision: {}\nrecall: {}\nf1 Score: {}".format(
            accuracy_score(_predict_test_label, _test_gt_vector),
            precision_score(_predict_test_label, _test_gt_vector, average=None),
            recall_score(_predict_test_label, _test_gt_vector, average=None),
            f1_score(_predict_test_label, _test_gt_vector, average=None)))
        print('memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print('time: {} s'.format(time.time() - _begin_time))


def aggregate_feature_vector(_data, _gt):
    # reshape ground truth
    _gt_vector = []
    for i_num in range(len(_gt)):
        _one_gt = np.full(shape=_data.shape[1], fill_value=_gt[i_num], dtype=int)
        _gt_vector = np.concatenate([_gt_vector, _one_gt])
    _gt_vector = np.array(_gt_vector)
    # extract temporal features from original data
    _data_vector = []
    for i_slice in _data:
        for j_slice in i_slice:
            j_slice_trans = j_slice.transpose([1, 0])
            extract_feature = []
            for k_slice in j_slice_trans:
                _max = np.max(k_slice)
                _min = np.min(k_slice)
                _var = np.std(k_slice)
                _slope, _intercept = np.nan, np.nan
                if np.isnan(k_slice).sum() == 0:
                    _slope, _intercept = np.polyfit(range(1, len(k_slice) + 1), k_slice, 1)
                _tmp_feature = [_max, _min, _var, _slope, _intercept]
                extract_feature.append(_tmp_feature)
            extract_feature_trans = np.array(extract_feature).transpose([1, 0])
            _data_vector.append(extract_feature_trans.ravel())
    _data_vector = np.array(_data_vector)

    # exclude pixels that have nan value
    indices = np.array(np.where(np.all(~np.isnan(_data_vector), axis=1)))
    _data_vector_r = _data_vector[indices[0], :]
    _gt_vector_r = _gt_vector[indices[0]]
    return _data_vector_r, _gt_vector_r


def aggregate_feature_vector_test(_data, _gt):
    # reshape ground truth
    _gt_vector = np.array(_gt).astype(int)
    # extract temporal features from original data
    _data_vector = []
    for j_slice in _data:
        j_slice_trans = j_slice.transpose([1, 0])
        extract_feature = []
        for k_slice in j_slice_trans:
            _max = np.max(k_slice)
            _min = np.min(k_slice)
            _var = np.std(k_slice)
            _slope, _intercept = np.nan, np.nan
            if np.isnan(k_slice).sum() == 0:
                _slope, _intercept = np.polyfit(range(1, len(k_slice) + 1), k_slice, 1)
            _tmp_feature = [_max, _min, _var, _slope, _intercept]
            extract_feature.append(_tmp_feature)
        extract_feature_trans = np.array(extract_feature).transpose([1, 0])
        _data_vector.append(extract_feature_trans.ravel())
    _data_vector = np.array(_data_vector)

    # exclude pixels that have nan value
    indices = np.array(np.where(np.all(~np.isnan(_data_vector), axis=1)))
    _data_vector_r = _data_vector[indices[0], :]
    _gt_vector_r = _gt_vector[indices[0]]
    return _data_vector_r, _gt_vector_r


def time_series_aggregate_features_classify(_train_data, _train_gt, _test_data, _test_gt):
    print("\nAlgorithm: time-series classification by aggregating features across time...")

    # data size: class_number * pixel_number * temporal_length * feature_length
    # ==> (class_number * pixel_number) * (temporal_feature_extraction*feature_length)
    _train_data_vector, _train_gt_vector = aggregate_feature_vector(_train_data, _train_gt)
    _test_data_vector, _test_gt_vector = aggregate_feature_vector_test(_test_data, _test_gt)
    print("Train data size: {}, {}, {}, {}".format(_train_data.shape, _train_gt.shape,
                                                   _train_data_vector.shape, _train_gt_vector.shape))
    print("test data size: {}, {}, {}, {}".format(_test_data.shape, _test_gt.shape,
                                                  _test_data_vector.shape, _test_gt_vector.shape))

    # classification
    classifier_name = ['Random forest', 'Support vector machine', 'Logistics regression', 'KNN neighborhood']
    classifier_func = [rf_classifier, svm_classifier, logistic_classifier, knn_classifier]
    for i in range(len(classifier_name)):
        print("\n{}:".format(classifier_name[i]))
        _begin_time = time.time()
        _predict_test_label = classifier_func[i](_train_data_vector, _train_gt_vector, _test_data_vector)
        print("accuracy: {}\nprecision: {}\nrecall: {}\nf1 Score: {}".format(
            accuracy_score(_predict_test_label, _test_gt_vector),
            precision_score(_predict_test_label, _test_gt_vector, average=None),
            recall_score(_predict_test_label, _test_gt_vector, average=None),
            f1_score(_predict_test_label, _test_gt_vector, average=None)))
        print('memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print('time: {} s'.format(time.time() - _begin_time))


def aggregate_prediction_vector(_data, _gt):
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
            _data_vector.append(j_slice)
    _data_vector = np.array(_data_vector)

    # exclude pixels that have nan value
    indices = np.array(np.where(np.all(~np.isnan(_data_vector), axis=(1, 2))))
    _data_vector_r = _data_vector[indices[0], :]
    _gt_vector_r = _gt_vector[indices[0]]
    return _data_vector_r, _gt_vector_r


def aggregate_prediction_vector_test(_data, _gt):
    # reshape ground truth
    _gt_vector = np.array(_gt).astype(int)
    # reshape data
    _data_vector = []
    for j_slice in _data:
        _data_vector.append(j_slice)
    _data_vector = np.array(_data_vector)

    # exclude pixels that have nan value
    indices = np.array(np.where(np.all(~np.isnan(_data_vector), axis=(1, 2))))
    _data_vector_r = _data_vector[indices[0], :]
    _gt_vector_r = _gt_vector[indices[0]]
    return _data_vector_r, _gt_vector_r


def time_series_aggregate_predictions_classify(_train_data, _train_gt, _test_data, _test_gt):
    print("\nAlgorithm: time-series classification by aggregating predictions across time...")

    # data size: class_number * pixel_number * temporal_length * feature_length
    # ==> (class_number * pixel_number) * temporal_length * feature_length
    _train_data_vector, _train_gt_vector = aggregate_prediction_vector(_train_data, _train_gt)
    _test_data_vector, _test_gt_vector = aggregate_prediction_vector_test(_test_data, _test_gt)
    print("Train data size: {}, {}, {}, {}".format(_train_data.shape, _train_gt.shape,
                                                   _train_data_vector.shape, _train_gt_vector.shape))
    print("test data size: {}, {}, {}, {}".format(_test_data.shape, _test_gt.shape,
                                                  _test_data_vector.shape, _test_gt_vector.shape))

    _train_data_vector = _train_data_vector.transpose([1, 0, 2])
    _test_data_vector = _test_data_vector.transpose([1, 0, 2])
    print("Transpose data size: {}, {}".format(_train_data_vector.shape, _test_data_vector.shape))

    # classification
    classifier_name = ['Random forest', 'Support vector machine', 'Logistics regression', 'KNN neighborhood']
    classifier_func = [rf_classifier, svm_classifier, logistic_classifier, knn_classifier]
    for i in range(len(classifier_name)):
        print("\n{}:".format(classifier_name[i]))
        _begin_time = time.time()
        # classify the label every time step
        _predict_out = []
        for time_slice in range(len(_train_data_vector)):
            _slice_train_data = _train_data_vector[time_slice]
            _slice_test_data = _test_data_vector[time_slice]
            _one_predict = classifier_func[i](_slice_train_data, _train_gt_vector, _slice_test_data)
            _predict_out.append(_one_predict)
        _predict_out = np.array(_predict_out).transpose([1, 0])
        # aggregate classification result by voting
        _final_predict = []
        for _one_slice in _predict_out:
            _one_slice = _one_slice.astype(int)
            _counts = np.bincount(_one_slice)
            _final_predict.append(np.argmax(_counts))
        # evaluation metrics
        print("accuracy: {}\nprecision: {}\nrecall: {}\nf1 Score: {}".format(
            accuracy_score(_final_predict, _test_gt_vector),
            precision_score(_final_predict, _test_gt_vector, average=None),
            recall_score(_final_predict, _test_gt_vector, average=None),
            f1_score(_final_predict, _test_gt_vector, average=None)))
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
    test_h5 = h5py.File(root + '/test_data.hdf5', 'r')
    test_data, test_GT = np.array(test_h5['DATA']), np.array(test_h5['GT'])
    # print("test data size: {}, {}".format(test_data.shape, test_GT.shape))
    print(np.array(train_GT))

    print('before inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    begin_time = time.time()

    # data size: class_number * pixel_number * temporal_length * feature_length
    time_series_stacking_classify(train_data, train_GT, test_data, test_GT)
    print('stacking inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print('stacking inference time: {}'.format(time.time() - begin_time))
    begin_time = time.time()

    time_series_aggregate_features_classify(train_data, train_GT, test_data, test_GT)
    print('aggregate feature inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print('aggregate feature inference time: {}'.format(time.time() - begin_time))
    begin_time = time.time()

    time_series_aggregate_predictions_classify(train_data, train_GT, test_data, test_GT)
    print('aggregate prediction inference, memory: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print('aggregate prediction inference time: {}'.format(time.time() - begin_time))
