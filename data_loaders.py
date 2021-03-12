"""Classes for loading dataset"""
# coding: utf-8

import os
import numpy as np


class BaseDataLoader:
    """Base data loader"""

    def __init__(self, dataset_dir):
        """Initiate data loader"""
        self.__dataset_dir = dataset_dir

    def get_data_train(self):
        """Get data for training"""
        raise NotImplementedError()

    def get_data_val(self):
        """Get data for validation"""
        raise NotImplementedError()

    def get_num_labels(self):
        """Get all labels"""
        raise NotImplementedError()

    def get_dataset_dir(self):
        """Get dataset directory"""
        return self.__dataset_dir


class NPZDataLoader(BaseDataLoader):
    """Load mnist.npz"""

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        with np.load(os.path.join(dataset_dir, 'mnist.npz')) as data:
            self.__data_train = list(zip(data['x_train'], data['y_train']))
            self.__data_val = list(zip(data['x_test'], data['y_test']))

    def get_data_train(self):
        return self.__data_train

    def get_data_val(self):
        return self.__data_val

    def get_num_labels(self):
        return 10

# from tqdm import tqdm
# from PIL import Image
# class MyBaseDataLoader(BaseDataLoader):
#     """从解压缩后的文件中读取数据"""
#
#     def get_data_train(self):
#         return self._read_data(os.path.join(self.get_dataset_dir(), 'MNIST/train'))
#
#     def get_data_val(self):
#         return self._read_data(os.path.join(self.get_dataset_dir(), 'MNIST/dev'))
#
#     def get_labels(self):
#         return list(range(10))
#
#     @classmethod
#     def _read_data(cls, path):
#         """读取MNIST数据集"""
#         images = []
#         for root, ds, fs in os.walk(os.path.join(path, 'images')):
#             for f in tqdm(fs, desc='Reading data from ' + path):
#                 image = Image.open(os.path.join(root, f))
#                 images.append(np.array(image, dtype=np.float32))
#         labels = np.load(os.path.join(path, 'labels.npy'))  # one-hot
#         labels = np.argmax(labels, 1)  # ids
#         return list(zip(images, labels))
#
#
# class TF1DataLoader(BaseDataLoader):
#     """从压缩文件中直接读取数据"""
#
#     def __init__(self, dataset_dir):
#         super(TF1DataLoader, self).__init__(dataset_dir)
#         import tensorflow.examples.tutorials.mnist.input_data as input_data  # for tensorflow 1.x
#         self.__mnist = input_data.read_data_sets(os.path.join(dataset_dir, 'MNIST/'))
#
#     def get_data_train(self):
#         data = []
#         for i in tqdm(range(len(self.__mnist.train.images))):
#             data.append([np.reshape(self.__mnist.train.images[i], (28, 28)) * 255,
#                          self.__mnist.train.labels[i]])
#         return data
#
#     def get_data_val(self):
#         data = []
#         for i in tqdm(range(len(self.__mnist.validation.images))):
#             data.append([np.reshape(self.__mnist.validation.images[i], (28, 28)) * 255,
#                          self.__mnist.validation.labels[i]])
#         return data
#
#     def get_labels(self):
#         return list(range(10))
