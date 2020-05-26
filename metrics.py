# coding: utf-8

import numpy as np


class Metrics:
    """根据混淆矩阵计算各项指标"""

    def __init__(self, labels):
        self.__confusion_matrix = np.zeros([len(labels), len(labels)])
        self.__label_dict = {}
        for (i, label) in enumerate(labels):
            self.__label_dict[label] = i

    def update(self, actual_labels, pred_labels):
        """根据真实标签和预测标签更新混淆矩阵"""
        if len(actual_labels) != len(pred_labels):
            raise ValueError('长度不同')
        for (a, p) in zip(actual_labels, pred_labels):
            a = self.__label_dict[a]
            p = self.__label_dict[p]
            self.__confusion_matrix[p][a] += 1

    def get_confusion_matrix(self):
        """混淆矩阵"""
        return self.__confusion_matrix.copy()

    def get_accuracy(self):
        """准确率"""
        return self.__confusion_matrix.trace() / self.__confusion_matrix.sum()