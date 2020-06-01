# coding: utf-8

import os
import numpy as np
from tqdm import tqdm
from PIL import Image


# import tensorflow.examples.tutorials.mnist.input_data as input_data  # for tensorflow 1.x


class DataLoader:
    """读取数据的基类，数据以列表形式存储"""

    def __init__(self, path):
        """定义数据集根目录"""
        self.__path = path

    def get_data_train(self):
        """获取训练集数据"""
        raise NotImplementedError()

    def get_data_dev(self):
        """获取开发集数据"""
        raise NotImplementedError()

    def get_data_test(self):
        """获取测试集数据"""
        raise NotImplementedError()

    def get_labels(self):
        """获取所有标签"""
        raise NotImplementedError()

    def get_path(self):
        return self.__path


class MyDataLoader(DataLoader):
    """从解压缩后的文件中读取数据"""

    def get_data_train(self):
        return self._read_data(os.path.join(self.get_path(), 'MNIST/train'))

    def get_data_dev(self):
        return self._read_data(os.path.join(self.get_path(), 'MNIST/dev'))

    def get_data_test(self):
        return self._read_data(os.path.join(self.get_path(), 'MNIST/test'))

    def get_labels(self):
        return list(range(10))

    @classmethod
    def _read_data(cls, path):
        """读取MNIST数据集"""
        images = []
        for root, ds, fs in os.walk(os.path.join(path, 'images')):
            for f in tqdm(fs, desc='Reading data from ' + path):
                image = Image.open(os.path.join(root, f))
                images.append(np.array(image, dtype=np.float32))
        labels = np.load(os.path.join(path, 'labels.npy'))  # one-hot
        labels = np.argmax(labels, 1)  # ids
        return list(zip(images, labels))


class TF1DataLoader(DataLoader):
    """从压缩文件中直接读取数据"""

    def __init__(self, path):
        super(TF1DataLoader, self).__init__(path)
        self.__mnist = input_data.read_data_sets(os.path.join(path, 'MNIST/'))

    def get_data_train(self):
        data = []
        for i in tqdm(range(len(self.__mnist.train.images))):
            data.append([np.reshape(self.__mnist.train.images[i], (28, 28)) * 255,
                         self.__mnist.train.labels[i]])
        return data

    def get_data_dev(self):
        data = []
        for i in tqdm(range(len(self.__mnist.validation.images))):
            data.append([np.reshape(self.__mnist.validation.images[i], (28, 28)) * 255,
                         self.__mnist.validation.labels[i]])
        return data

    def get_data_test(self):
        data = []
        for i in tqdm(range(len(self.__mnist.test.images))):
            data.append([np.reshape(self.__mnist.test.images[i], (28, 28)) * 255,
                         self.__mnist.test.labels[i]])
        return data

    def get_labels(self):
        return list(range(10))


class TF2DataLoader(DataLoader):
    """读取以npz格式存储的MNIST"""

    def __init__(self, path, train_prop=0.8):
        super(TF2DataLoader, self).__init__(path)
        with np.load(os.path.join(path, 'MNIST', 'mnist.npz')) as f:
            self.__data_train = list(zip(f['x_train'], f['y_train']))
            self.__data_test = list(zip(f['x_test'], f['y_test']))
        split_index = int(train_prop * len(self.__data_train))
        self.__data_dev = self.__data_train[split_index:]
        self.__data_train = self.__data_train[:split_index]

    def get_data_train(self):
        return self.__data_train

    def get_data_dev(self):
        return self.__data_dev

    def get_data_test(self):
        return self.__data_test

    def get_labels(self):
        return list(range(10))
