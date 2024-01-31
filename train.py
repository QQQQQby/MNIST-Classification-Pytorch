"""Train CNN model on MNIST"""

import random
import time
import os
import argparse

import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import data_loaders
import modules
from metrics import MetricsCalculator


def train(args):
    """Train"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    model = modules.MyResNet50()
    # model = modules.CNN1()

    # Read data
    print('Reading data...', flush=True)
    data_loader = data_loaders.NPZDataLoader(args.dataset_dir)
    data_train = data_loader.get_data_train()
    data_val = data_loader.get_data_val()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare to train
    model.train()
    train_acc_history, val_acc_history = [], []
    loss_history = []
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Start training
    print('Start training.', flush=True)
    for epoch in range(args.num_epochs):

        # Train
        start_time = time.time()

        random.shuffle(data_train)
        calculator = MetricsCalculator(10)
        for start in tqdm(range(0, len(data_train), args.train_batch_size),
                          desc=f'Training epoch {epoch}: '):
            images = np.array([data_train[idx][0] for idx in
                               range(start, min(start + args.train_batch_size, len(data_train)))])
            actual_labels = [data_train[idx][1] for idx in
                             range(start, min(start + args.train_batch_size, len(data_train)))]

            # forward
            outputs = model(torch.tensor(images, dtype=torch.float32))

            # backward
            batch_labels = torch.tensor(actual_labels, dtype=torch.int64)
            model.zero_grad()
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            loss_history.append(float(loss))
            loss.backward()
            optimizer.step()

            pred_labels = outputs.softmax(1).argmax(1).tolist()
            calculator.update(actual_labels, pred_labels)
        acc = calculator.calc_accuracy()
        print(f'Accuracy: {acc}')
        train_acc_history.append(acc)

        end_time = time.time()
        print(f'Training lasts {end_time - start_time} s')

        if args.output_dir:
            torch.save(model, os.path.join(args.output_dir, 'epoch_' + str(epoch) + '.pt'))

        if args.not_val:
            continue

        # Validate
        model.eval()
        start_time = time.time()

        calculator = MetricsCalculator(10)
        for start in tqdm(range(0, len(data_val), args.val_batch_size),
                          desc=f'Validating epoch {epoch}: '):
            images = np.array([data_val[idx][0] for idx in
                               range(start, min(start + args.val_batch_size, len(data_val)))])
            actual_labels = [data_val[idx][1] for idx in
                             range(start, min(start + args.val_batch_size, len(data_val)))]

            # forward
            outputs = model(torch.tensor(images, dtype=torch.float32))

            # Update metrics
            pred_labels = outputs.softmax(1).argmax(1).tolist()
            calculator.update(actual_labels, pred_labels)
        acc = calculator.calc_accuracy()
        print(f'Accuracy: {acc}')
        val_acc_history.append(acc)

        end_time = time.time()
        print(f'Validating lasts {end_time - start_time} s')
        print()

    # Plot
    if args.output_dir:
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.grid(True)
        plt.plot(loss_history, 'r')
        plt.savefig(os.path.join(args.output_dir, 'loss.jpg'))
        plt.close()

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy During Training')
        plt.grid(True)
        plt.plot(train_acc_history, 'r')
        plt.plot(val_acc_history, 'b')
        plt.legend(['train', 'val'])
        plt.savefig(os.path.join(args.output_dir, 'acc.jpg'))
        plt.close()


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Train MNIST Classifier.")
    parser.add_argument('--dataset_dir', type=str, default='./data/MNIST',
                        help='The directory where the dataset is located.')
    parser.add_argument('--output_dir', type=str, default='./output/resnet2/',
                        help='Output directory.')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='Batch size of train set.')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--not_val', action='store_true', default=False,
                        help="Whether not to validate the model.")
    parser.add_argument('--val_batch_size', type=int, default=1024,
                        help='Batch size of validation set.')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
