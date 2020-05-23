# coding: utf-8
"""
Pytorch modules
input: 28*28, output(one-hot): 10
"""

import torch
from torch import nn, optim


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
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(300, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 10, bias=True)
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        # self.op = optim.SGD(self.parameters(), lr=0.001, momentum=0.8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
