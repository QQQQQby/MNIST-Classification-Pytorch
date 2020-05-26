# Pytorch Implement of Image Classification for MNIST

## Dataset

We use MNIST dataset. There are 2 data-loaders in `data_loaders.py`. If you want to use `MyDataLoader`, you should run the following command to firstly extract images and labels from MNIST:

```
python preprocess.py
```

However, we found that doing so will slow down the loading of data before running the model. So you'd better use `MNISTDataLoader` instead.


## Run

```
optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset path.
  --not_train           Whether not to train the model.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size of train set.
  --epochs EPOCHS       Number of epochs.
  --lr LR               Learning rate.
  --not_eval            Whether not to evaluate the model.
  --dev_batch_size DEV_BATCH_SIZE
                        Batch size of dev set.
  --not_test            Whether not to test the model.
  --test_batch_size TEST_BATCH_SIZE
                        Batch size of test set.
```

Train, test and evaluate:

```
python classifier.py
```
