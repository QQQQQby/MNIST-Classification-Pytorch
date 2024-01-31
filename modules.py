# coding: utf-8
"""Pytorch modules to classify MNIST"""
import torchvision

from torch import nn


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 * 28 * 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 32 * 14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, 10, bias=True)
        )
        # 10

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MyResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 3, 5, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 10, bias=True)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.preprocess(x)
        x = self.backbone(x)
        return x


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 * 28 * 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 32 * 14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(1024, 100, bias=True),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(100, 10, bias=True)
        )
        # 10

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
