"""Calculate metrics"""
# coding: utf-8

import numpy as np


class MetricsCalculator:
    """Used for calculating metrics according to the confusion matrix"""

    def __init__(self, num_labels):
        self.confusion_matrix = np.zeros([num_labels, num_labels])

    def update(self, actual_labels, pred_labels):
        """Update confusion matrix"""
        if len(actual_labels) != len(pred_labels):
            raise ValueError('长度不同')
        for actual_label, pred_label in zip(actual_labels, pred_labels):
            self.confusion_matrix[pred_label][actual_label] += 1

    def calc_accuracy(self):
        """Calculate accuracy"""
        return self.confusion_matrix.trace() / self.confusion_matrix.sum()
