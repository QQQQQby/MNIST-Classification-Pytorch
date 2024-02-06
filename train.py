"""
Train model on MNIST
"""

import os
import time

import torch
from torch import nn
from torchvision import datasets
import numpy as np
import click
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import MyResNet18
from utils import shuffle_arrays_in_unison


@click.command()
@click.option('-t', '--train_prop', type=float, default=0.8,
              help='Proportion of train set to the total dataset.')
@click.option('-e', '--epochs', type=int, default=100,
              help='Number of training epochs.')
@click.option('-p', '--patience', type=int, default=10,
              help='If the model performs poorly for a specified number '
                   'of consecutive epochs, training will be stopped.')
@click.option('-b', '--batch_size', type=int, default=256,
              help='Batch size.')
@click.option('-l', '--lr0', type=float, default=0.01,
              help='Initial learning rate.')
@click.option('-m', '--momentum', type=float, default=0.9,
              help='Momentum of SGD.')
@click.option('-o', '--out_dir', type=str, default=None,
              help='Training output directory.')
@click.option('--batch_size_val', type=int, default=None,
              help='Validating batch size. If not set, it will be equal to batch_size.')
def train(train_prop, epochs, patience, batch_size, lr0, momentum, out_dir, batch_size_val):
    """Train on MNIST"""
    # Prepare output directory
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_root = './output/'
        out_dir = os.path.join(out_root, 'train')
        if os.path.isdir(out_dir):
            i = 1
            while True:
                out_dir = os.path.join(out_root, f'train_{i}')
                if not os.path.isdir(out_dir):
                    break
                i += 1
        os.makedirs(out_dir)
    print(f'Results will be saved in \"{out_dir}\"')

    # Download and read the MNIST dataset
    print('Reading data...', flush=True)
    mnist = datasets.MNIST('./datasets/', download=True, train=True)
    images_train, labels_train = np.copy(mnist.data), np.copy(mnist.targets)
    shuffle_arrays_in_unison(images_train, labels_train)

    # Divide the dataset into train set and val set
    num_data_train = int(images_train.shape[0] * train_prop)
    num_data_val = images_train.shape[0] - num_data_train
    images_val, labels_val = images_train[num_data_train:], labels_train[num_data_train:]
    images_train, labels_train = images_train[:num_data_train], labels_train[:num_data_train]

    # Set default device to CUDA if available
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = MyResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum)

    # Prepare to train
    acc_history_train, loss_history, acc_history_val = [], [], []
    max_acc_val, max_acc_val_epoch, curr_patience = 0, 0, patience
    early_stopped = False
    if not batch_size_val:
        batch_size_val = batch_size

    # Start training
    print('Start training.')
    print(flush=True)
    for epoch in range(epochs):

        # Training process
        start_time = time.perf_counter_ns()

        model.train()
        shuffle_arrays_in_unison(images_train, labels_train)
        num_correct = 0

        for start in tqdm(range(0, images_train.shape[0], batch_size),
                          desc=f'Training epoch {epoch}: '):
            # Get batch data
            batch_images = images_train[start: start + batch_size]
            batch_labels = labels_train[start: start + batch_size]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(torch.tensor(batch_images, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(batch_labels))
            loss.backward()
            optimizer.step()

            loss_history.append(float(loss))

            # Update accuracy
            pred_labels = outputs.argmax(1).cpu().numpy()
            num_correct += np.sum(batch_labels == pred_labels)

        acc_train = num_correct / num_data_train
        acc_history_train.append(acc_train)
        end_time = time.perf_counter_ns()
        print(f'Accuracy: {acc_train}')
        print(f'Time: {(end_time - start_time) / 1e9} s.', flush=True)

        torch.save(model, os.path.join(out_dir, 'last.pt'))

        # Validating process
        start_time = time.perf_counter_ns()

        model.eval()
        num_correct = 0

        with torch.no_grad():
            for start in tqdm(range(0, len(images_val), batch_size_val),
                              desc=f'Validating epoch {epoch}: '):
                # Get batch data
                batch_images = images_val[start: start + batch_size_val]
                batch_labels = labels_val[start: start + batch_size_val]

                # Inference
                outputs = model(torch.tensor(batch_images, dtype=torch.float32))

                # Update accuracy
                pred_labels = outputs.argmax(1).cpu().numpy()
                num_correct += np.sum(batch_labels == pred_labels)

        acc_val = num_correct / num_data_val
        acc_history_val.append(acc_val)
        end_time = time.perf_counter_ns()
        print(f'Accuracy: {acc_val}')
        print(f'Time: {(end_time - start_time) / 1e9} s.')
        print(flush=True)

        if acc_val > max_acc_val:
            max_acc_val = acc_val
            max_acc_val_epoch = epoch

            torch.save(model, os.path.join(out_dir, 'best.pt'))
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience < 0:
                early_stopped = True
                break

    if early_stopped:
        print('The model\'s performance reaches its best after being trained '
              f'for {max_acc_val_epoch} epochs, so training is stopped early.')
    else:
        print('Training is completed.', flush=True)

    # Plot result and save
    plt.figure(figsize=(16, 8))

    plt.subplot(121)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.plot(loss_history, 'r')

    plt.subplot(122)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.ylim(0, 1)
    plt.yticks(np.append(np.arange(0, 1, 0.05), 1))
    plt.grid(True)
    plt.plot(acc_history_train, 'r')
    plt.plot(acc_history_val, 'b')
    plt.legend(['train', 'val'])

    plt.savefig(os.path.join(out_dir, 'result.jpg'))
    plt.close()


if __name__ == '__main__':
    train()
