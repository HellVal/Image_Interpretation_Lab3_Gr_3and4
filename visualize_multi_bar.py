"""
Created on Wed Dec 15 15:11:21 2021
@author: Yatao
@purpose: visualization for accuracy assessment
"""
import numpy as np
import matplotlib.pyplot as plt


def vis_multi_bar():
    labels_method = ['m_1', 'm_2', 'm_3']
    # labels_method = ['Stacking pixels', 'Aggregate features', 'Aggregate predictions']
    labels_classifier = ['RF', 'SVM', 'KNN']

    rf_acc = [0.9314,0.9249,0.7967]
    rf_pre = [0.9341,0.9278,0.8007]
    rf_rec = [0.9345,0.9283,0.8087]
    rf_f1c = [0.9340,0.9276,0.7977]

    svm_acc = [0.8329,0.8154,0.4924]
    svm_pre = [0.8366,0.8186,0.4877]
    svm_rec = [0.8374,0.8203,0.6104]
    svm_f1c = [0.8366,0.8183,0.4484]

    knn_acc = [0.9334,0.9243,0.5638]
    knn_pre = [0.9363,0.9273,0.5639]
    knn_rec = [0.9353,0.9267,0.7779]
    knn_f1c = [0.9347,0.9261,0.5765]

    plt.figure(figsize=(14, 4))

    # random forest
    plt.subplot(131)
    x = np.arange(len(labels_method))
    width = 0.12
    plt.bar(x - 1.5 * width, rf_acc, width, color='coral', label='Overall accuracy')
    plt.bar(x - 0.5 * width, rf_pre, width, color='orange', label='Average precision')
    plt.bar(x + 0.5 * width, rf_rec, width, color='mediumseagreen', label='Average recall')
    plt.bar(x + 1.5 * width, rf_f1c, width, color='cornflowerblue', label='Average F1 score')
    plt.ylabel('Performance')
    plt.title('RF', y=-0.16)
    plt.xticks(x, labels=labels_method)

    # support vector machine
    plt.subplot(132)
    x = np.arange(len(labels_method))
    plt.bar(x - 1.5 * width, svm_acc, width, color='coral', label='Overall accuracy')
    plt.bar(x - 0.5 * width, svm_pre, width, color='orange', label='Average precision')
    plt.bar(x + 0.5 * width, svm_rec, width, color='mediumseagreen', label='Average recall')
    plt.bar(x + 1.5 * width, svm_f1c, width, color='cornflowerblue', label='Average F1 score')
    plt.ylabel('Performance')
    plt.title('SVM', y=-0.16)
    plt.xticks(x, labels=labels_method)
    plt.legend(bbox_to_anchor=(0.5, 1.02), ncol=4, loc='lower center')

    # knn
    plt.subplot(133)
    x = np.arange(len(labels_method))
    plt.bar(x - 1.5 * width, knn_acc, width, color='coral', label='Overall accuracy')
    plt.bar(x - 0.5 * width, knn_pre, width, color='orange', label='Average precision')
    plt.bar(x + 0.5 * width, knn_rec, width, color='mediumseagreen', label='Average recall')
    plt.bar(x + 1.5 * width, knn_f1c, width, color='cornflowerblue', label='Average F1 score')
    plt.ylabel('Performance')
    plt.title('KNN', y=-0.16)
    plt.xticks(x, labels=labels_method)
    plt.show()


if __name__ == '__main__':
    print('test')
    vis_multi_bar()
