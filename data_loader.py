# coding: utf-8
import random
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


class DataLoader:
    """读取数据的基类，数据以列表形式存储"""

    def __init__(self, train_path, dev_path, test_path):
        self.__data_train = self.__read_data_train(train_path)
        self.__data_dev = self.__read_data_dev(dev_path)
        self.__data_test = self.__read_data_test(test_path)

    def __read_data_train(self, path):
        """读取训练集数据"""
        raise NotImplementedError()

    def __read_data_dev(self, path):
        """读取开发集数据"""
        raise NotImplementedError()

    def __read_data_test(self, path):
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


    # 返回下一个batch的数据
    def get_next_batch(self):
        if self.__present_batch >= len(self.__data):
            self.__present_batch = 0
        res = self.__data[self.__present_batch:
                          self.__present_batch + self.__batch_size]
        self.__present_batch += self.__batch_size
        return res

    # 计算还需要迭代的次数
    def get_num_steps(self):
        return (len(self.__data) - self.__present_batch - 1) // self.__batch_size + 1

    # 重新设置批次
    def restore_batch(self):
        self.__present_batch = 0

    def __iter__(self):
        if self.__present_batch >= len(self.__data):
            self.__present_batch = 0
        for i in range(self.get_num_steps()):
            yield self.get_next_batch()

    @classmethod
    def _read_MNIST_data(cls, path):
        images = []
        for root, ds, fs in os.walk(os.path.join(path, 'images')):
            for f in tqdm(fs, desc='Reading data from ' + path):
                image = Image.open(os.path.join(root, f))
                images.append(np.array(image))
        labels = np.load(os.path.join(path, 'labels.npy'))
        return list(zip(images, labels))

# if __name__ == '__main__':
#     loader = DataLoader([1, 2, 3, 4, 5, 6, 7], batch_size=4)
#     loader.shuffle_data()
#     print(loader.get_data_copy())
#     print()
#     print(loader.get_next_batch())
#     print(loader.get_next_batch())
#     for batch in loader:
#         print(batch)
