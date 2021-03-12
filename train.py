"""Train CNN model on MNIST"""
# coding: utf-8
# pylint: disable=no-member, not-callable

import random
import time
import os
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm

import data_loaders
from metrics import MetricsCalculator
import modules


def train(args):
    """Train"""
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = modules.CNN1()

    # Read data
    print('Reading data...', flush=True)
    data_loader = data_loaders.NPZDataLoader(args.dataset_dir)
    data_train = data_loader.get_data_train()
    data_val = data_loader.get_data_val()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare to train
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Start training
    print('Start training.', flush=True)
    for epoch in range(args.num_epochs):

        # Train
        start_time = time.time()

        random.shuffle(data_train)
        calculator = MetricsCalculator(10)
        for start in tqdm(range(0, len(data_train), args.train_batch_size),
                          desc='Training epoch %d: ' % epoch):
            images = [data_train[idx][0] for idx in
                      range(start, min(start + args.train_batch_size, len(data_train)))]
            actual_labels = [data_train[idx][1] for idx in
                             range(start, min(start + args.train_batch_size, len(data_train)))]

            # forward
            outputs = model(torch.tensor(images, dtype=torch.float32))

            # backward
            batch_labels = torch.tensor(actual_labels, dtype=torch.int64)
            model.zero_grad()
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            pred_labels = outputs.softmax(1).argmax(1).tolist()
            calculator.update(actual_labels, pred_labels)
        print('Accuracy:', calculator.calc_accuracy())

        end_time = time.time()
        print('Training lasts', end_time - start_time, 's')

        if args.output_dir:
            torch.save(model, os.path.join(args.output_dir, 'epoch_' + str(epoch) + '.pt'))

        if args.not_val:
            continue

        # Validate
        start_time = time.time()

        calculator = MetricsCalculator(10)
        # m = metrics.Metrics(labels)
        for start in tqdm(range(0, len(data_val), args.val_batch_size),
                          desc='Validating epoch %d: ' % epoch):
            images = [data_val[idx][0] for idx in
                      range(start, min(start + args.val_batch_size, len(data_val)))]
            actual_labels = [data_val[idx][1] for idx in
                             range(start, min(start + args.val_batch_size, len(data_val)))]

            # forward
            outputs = model(torch.tensor(images, dtype=torch.float32))

            # Update metrics
            pred_labels = outputs.softmax(1).argmax(1).tolist()
            calculator.update(actual_labels, pred_labels)
        print('Accuracy:', calculator.calc_accuracy())

        end_time = time.time()
        print('Validating lasts', end_time - start_time, 's')


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Train MNIST Classifier.")
    parser.add_argument('--dataset_dir', type=str, default='./data/MNIST',
                        help='The directory where the dataset is located.')
    parser.add_argument('--output_dir', type=str, default='./output/1/',
                        help='Output directory.')
    parser.add_argument('--train_batch_size', type=int, default=500,
                        help='Batch size of train set.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--not_val', action='store_true', default=False,
                        help="Whether not to validate the model.")
    parser.add_argument('--val_batch_size', type=int, default=500,
                        help='Batch size of validation set.')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
