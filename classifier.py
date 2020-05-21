# coding: utf-8

import data_loader
import models

import argparse
import os


class Classifier:
    def __init__(self, model, args):
        print(args)
        self.model = model
        self.args = args
        self.data_train_loader = data_loader.DataLoader(os.path.join(args.dataset_path, 'train/'),
                                                        args.train_batch_size)
        self.data_dev_loader = data_loader.DataLoader(os.path.join(args.dataset_path, 'dev/'),
                                                      args.dev_batch_size)
        self.data_test_loader = data_loader.DataLoader(os.path.join(args.dataset_path, 'test/'),
                                                       args.test_batch_size)

    def run(self):
        if not self.args.not_train:
            self.train()
        if not self.args.not_eval:
            self.eval()
        if not self.args.not_test:
            self.test()

    def train(self):
        pass
        # for epoch in range(self.args.epochs):


    def eval(self):
        pass

    def test(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST Classifier.")
    parser.add_argument('--dataset_path', type=str, default='./data/MNIST/',
                        help='Dataset path.')

    parser.add_argument('--not_train', action='store_false', default=True,
                        help="Whether not to train the model.")
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs.')
    parser.add_argument('--train_batch_size', type=int, default=500,
                        help='Batch size of train set.')

    parser.add_argument('--not_eval', action='store_false', default=True,
                        help="Whether not to evaluate the model.")
    parser.add_argument('--dev_batch_size', type=int, default=1000,
                        help='Batch size of dev set.')

    parser.add_argument('--not_test', action='store_false', default=True,
                        help="Whether not to test the model.")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='Batch size of test set.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    classifier = Classifier(models.MyCNN(), parse_args())
