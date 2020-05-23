# coding: utf-8
import random
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


class DataLoader:
    """读取数据的基类，数据以列表形式存储"""

    def __init__(self, train_path, dev_path, test_path):
        self.__data_train = self._read_data_train(train_path)
        self.__data_dev = self._read_data_dev(dev_path)
        self.__data_test = self._read_data_test(test_path)

    def _read_data_train(self, path):
        """读取训练集数据"""
        raise NotImplementedError()

    def _read_data_dev(self, path):
        """读取开发集数据"""
        raise NotImplementedError()

    def _read_data_test(self, path):
        """读取测试集数据"""
        raise NotImplementedError()

    def shuffle_data_train(self):
        """打乱训练集数据"""
        random.shuffle(self.__data_train)

    def get_data_train(self):
        """获取训练集数据"""
        return self.__data_train

    def get_data_dev(self):
        """获取开发集数据"""
        return self.__data_dev

    def get_data_test(self):
        """获取测试集数据"""
        return self.__data_test


class MyDataLoader(DataLoader):
    def _read_data_train(self, path):
        return self._read_MNIST_data(path)

    def _read_data_dev(self, path):
        return self._read_MNIST_data(path)

    def _read_data_test(self, path):
        return self._read_MNIST_data(path)

    @classmethod
    def _read_MNIST_data(cls, path):
        """读取MNIST数据集"""
        images = []
        for root, ds, fs in os.walk(os.path.join(path, 'images')):
            for f in tqdm(fs, desc='Reading data from ' + path):
                image = Image.open(os.path.join(root, f))
                images.append(np.array(image))
        labels = np.load(os.path.join(path, 'labels.npy'))
        return list(zip(images, labels))

    """    
        def get_next_batch(self):
            if self.__present_batch >= len(self.__data):
                self.__present_batch = 0
            res = self.__data[self.__present_batch:
                              self.__present_batch + self.__batch_size]
            self.__present_batch += self.__batch_size
            return res
            """


if __name__ == '__main__':
    loader = MyDataLoader('./data/MNIST/train', './data/MNIST/dev', './data/MNIST/test')
