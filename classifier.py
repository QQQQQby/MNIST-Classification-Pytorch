# coding: utf-8
import random

import data_loaders
import modules

import torch
from torch import nn, optim
import argparse
import os
from tqdm import tqdm


class Classifier:
    def __init__(self, model, args):
        print(args)
        self.model = model
        self.args = args
        data_loader = data_loaders.MNISTDataLoader(args.dataset_path)
        self.data_train = data_loader.get_data_train()
        self.data_dev = data_loader.get_data_dev()
        self.data_test = data_loader.get_data_test()

    def run(self):
        for epoch in range(self.args.epochs):
            if not self.args.not_train:
                """Train"""
                print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20)
                random.shuffle(self.data_train)
                for start in tqdm(range(0, len(self.data_train), self.args.train_batch_size), desc='Training batch: '):
                    batch_images = torch.tensor(
                        [d[0] for d in self.data_train[start:start + self.args.train_batch_size]],
                        dtype=torch.float32
                    ).unsqueeze(1)
                    batch_labels = torch.tensor(
                        [d[1] for d in self.data_train[start:start + self.args.train_batch_size]],
                        dtype=torch.int64
                    )
                    outputs = self.model(batch_images)
                    """backward"""
                    optimizer = optim.SGD(self.model.parameters(), lr=0.01)
                    self.model.zero_grad()
                    loss = nn.CrossEntropyLoss()(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

            if not self.args.not_eval:
                """Eval"""
                print('-' * 20 + 'Evaluating epoch %d' % epoch + '-' * 20)
                for start in tqdm(range(0, len(self.data_dev), self.args.train_batch_size), desc='Training batch: '):
                    batch_images = torch.tensor(
                        [d[0] for d in self.data_dev[start:start + self.args.train_batch_size]],
                        dtype=torch.float32
                    ).unsqueeze(1)
                    batch_labels = torch.tensor(
                        [d[1] for d in self.data_dev[start:start + self.args.train_batch_size]],
                        dtype=torch.int64
                    )
                    outputs = self.model(batch_images)
                    """evaluating"""

            if not self.args.not_test:
                """Test"""
                print('-' * 20 + 'Testing epoch %d' % epoch + '-' * 20)
                for start in tqdm(range(0, len(self.data_test), self.args.train_batch_size), desc='Training batch: '):
                    batch_images = torch.tensor(
                        [d[0] for d in self.data_test[start:start + self.args.train_batch_size]],
                        dtype=torch.float32
                    ).unsqueeze(1)
                    batch_labels = torch.tensor(
                        [d[1] for d in self.data_test[start:start + self.args.train_batch_size]],
                        dtype=torch.int64
                    )
                    """evaluating"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST Classifier.")
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Dataset path.')
    parser.add_argument('--not_train', action='store_true', default=False,
                        help="Whether not to train the model.")
    parser.add_argument('--train_batch_size', type=int, default=500,
                        help='Batch size of train set.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs.')

    parser.add_argument('--not_eval', action='store_true', default=False,
                        help="Whether not to evaluate the model.")
    parser.add_argument('--dev_batch_size', type=int, default=1000,
                        help='Batch size of dev set.')

    parser.add_argument('--not_test', action='store_true', default=False,
                        help="Whether not to test the model.")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='Batch size of test set.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    classifier = Classifier(modules.CNN(), parse_args())
    classifier.run()
