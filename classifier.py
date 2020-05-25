# coding: utf-8

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
        self.data_loader = data_loaders.MyDataLoader(train_path=args.train_path,
                                                     dev_path=args.dev_path,
                                                     test_path=args.test_path)

    def run(self):
        for epoch in range(self.args.epochs):
            if not self.args.not_train:
                """Train"""
                print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20)
                self.data_loader.shuffle_data_train()
                data = self.data_loader.get_data_train()
                for start in tqdm(range(0, len(data), self.args.train_batch_size), desc='Training batch: '):
                    batch_images = torch.tensor([d[0] for d in data[start:start + self.args.train_batch_size]],
                                                dtype=torch.float32).unsqueeze(1)
                    batch_labels = torch.tensor([d[1] for d in data[start:start + self.args.train_batch_size]],
                                                dtype=torch.int64)
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
                data = self.data_loader.get_data_dev()
                for start in tqdm(range(0, len(data), self.args.dev_batch_size), desc='Evaluating batch: '):
                    batch_images = torch.tensor([d[0] for d in data[start:start + self.args.dev_batch_size]],
                                                dtype=torch.float32).unsqueeze(1)
                    batch_labels = torch.tensor([d[1] for d in data[start:start + self.args.dev_batch_size]],
                                                dtype=torch.int64)
                    outputs = self.model(batch_images)
                    """evaluating"""


            if not self.args.not_test:
                """Test"""
                print('-' * 20 + 'Testing epoch %d' % epoch + '-' * 20)
                data = self.data_loader.get_data_dev()
                for start in tqdm(range(0, len(data), self.args.dev_batch_size), desc='Evaluating batch: '):
                    batch_images = torch.tensor([d[0] for d in data[start:start + self.args.dev_batch_size]],
                                                dtype=torch.float32).unsqueeze(1)
                    batch_labels = torch.tensor([d[1] for d in data[start:start + self.args.dev_batch_size]],
                                                dtype=torch.int64)
                    outputs = self.model(batch_images)
                    """evaluating"""



def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST Classifier.")
    parser.add_argument('--train_path', type=str, default='./data/MNIST/train',
                        help='Train set path.')
    parser.add_argument('--not_train', action='store_true', default=False,
                        help="Whether not to train the model.")
    parser.add_argument('--train_batch_size', type=int, default=500,
                        help='Batch size of train set.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs.')

    parser.add_argument('--dev_path', type=str, default='./data/MNIST/dev',
                        help='Dev set path.')
    parser.add_argument('--not_eval', action='store_true', default=False,
                        help="Whether not to evaluate the model.")
    parser.add_argument('--dev_batch_size', type=int, default=1000,
                        help='Batch size of dev set.')

    parser.add_argument('--test_path', type=str, default='./data/MNIST/test',
                        help='Test set path.')
    parser.add_argument('--not_test', action='store_true', default=False,
                        help="Whether not to test the model.")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='Batch size of test set.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    classifier = Classifier(modules.CNN(), parse_args())
    classifier.run()
