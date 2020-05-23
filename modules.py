# coding: utf-8
"""
Pytorch modules
input: 28*28, output(one-hot): 10
"""
from time import time

import torch
from torch import nn


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        """28 * 28 * 1"""
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1),
            nn.ReLU(),
            nn.Conv2d(6, 6, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """12 * 12 * 6"""
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 2, 1),
            nn.ReLU(),
            nn.Conv2d(12, 12, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """5 * 5 * 12"""
        self.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(300, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True)
        )
        """10"""

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    m = MyCNN()
    torch.manual_seed(1)
    print(m(torch.randn(5, 1, 28, 28)))
