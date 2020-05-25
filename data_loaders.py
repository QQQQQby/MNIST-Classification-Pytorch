# coding: utf-8

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow.examples.tutorials.mnist.input_data as input_data


class DataLoader:
    """读取数据的基类，数据以列表形式存储"""

    def __init__(self, path):
        """定义数据集根目录"""
        self.path = path

    def get_data_train(self):
        """获取训练集数据"""
        raise NotImplementedError()

    def get_data_dev(self):
        """获取开发集数据"""
        raise NotImplementedError()

    def get_data_test(self):
        """获取测试集数据"""
        raise NotImplementedError()


class MyDataLoader(DataLoader):
    """从解压缩后的文件中读取数据"""

    def get_data_train(self):
        return self._read_data(os.path.join(self.path, 'MNIST/train'))

    def get_data_dev(self):
        return self._read_data(os.path.join(self.path, 'MNIST/dev'))

    def get_data_test(self):
        return self._read_data(os.path.join(self.path, 'MNIST/test'))

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


class MNISTDataLoader(DataLoader):
    """从压缩文件中直接读取数据"""

    def __init__(self, path):
        super(MNISTDataLoader, self).__init__(path)
        self.mnist = input_data.read_data_sets(os.path.join(path, 'MNIST/'), one_hot=True)

    def get_data_train(self):
        data = []
        for i in tqdm(range(len(self.mnist.train.images))):
            data.append([np.reshape(self.mnist.train.images[i], (28, 28)) * 255,
                         np.argmax(self.mnist.train.labels[i])])
        return data

    def get_data_dev(self):
        data = []
        for i in tqdm(range(len(self.mnist.validation.images))):
            data.append([np.reshape(self.mnist.validation.images[i], (28, 28)) * 255,
                         np.argmax(self.mnist.validation.labels[i])])
        return data

    def get_data_test(self):
        data = []
        for i in tqdm(range(len(self.mnist.test.images))):
            data.append([np.reshape(self.mnist.test.images[i], (28, 28)) * 255,
                         np.argmax(self.mnist.test.labels[i])])
        return data
