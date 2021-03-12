# Pytorch Implement of Image Classification for MNIST

## How to run

### Train

```
usage: train.py [-h] [--dataset_dir DATASET_DIR] [--output_dir OUTPUT_DIR]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--lr LR] [--not_val]
                [--val_batch_size VAL_BATCH_SIZE]

Train MNIST Classifier.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        The directory where the dataset is located.
  --output_dir OUTPUT_DIR
                        Output directory.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size of train set.
  --num_epochs NUM_EPOCHS
                        Number of epochs.
  --lr LR               Learning rate.
  --not_val             Whether not to validate the model.
  --val_batch_size VAL_BATCH_SIZE
                        Batch size of validation set.

```
