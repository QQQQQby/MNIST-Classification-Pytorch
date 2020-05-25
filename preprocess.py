# coding: utf-8
# 读取MNIST数据集，将图片和标签提取出来
import os
from tqdm import tqdm

import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image

if __name__ == '__main__':
    mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
    for path in ['./data/MNIST/train/images', './data/MNIST/dev/images', './data/MNIST/test/images']:
        if not os.path.exists(path):
            os.makedirs(path)

    for i in tqdm(range(len(mnist.train.images))):
        im_data = np.array(np.reshape(mnist.train.images[i], (28, 28)) * 255, dtype=np.uint8)
        img = Image.fromarray(im_data, 'L')
        img.save('data/MNIST/train/images/' + str(i) + '.jpg')
    for i in tqdm(range(len(mnist.validation.images))):
        im_data = np.array(np.reshape(mnist.validation.images[i], (28, 28)) * 255, dtype=np.uint8)
        img = Image.fromarray(im_data, 'L')
        img.save('data/MNIST/dev/images/' + str(i) + '.jpg')
    for i in tqdm(range(len(mnist.test.images))):
        im_data = np.array(np.reshape(mnist.test.images[i], (28, 28)) * 255, dtype=np.uint8)
        img = Image.fromarray(im_data, 'L')
        img.save('data/MNIST/test/images/' + str(i) + '.jpg')

    np.save('data/MNIST/train/labels', mnist.train.labels)
    np.save('data/MNIST/dev/labels', mnist.validation.labels)
    np.save('data/MNIST/test/labels', mnist.test.labels)