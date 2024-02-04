# PyTorch Implement of Image Classification for MNIST

## How to run

### Train

```
Usage: train.py [OPTIONS]

  Train on MNIST

Options:
  -t, --train_prop FLOAT    Proportion of train set to the total dataset.
  -e, --epochs INTEGER      Number of training epochs.
  -p, --patience INTEGER    If the model performs poorly for a specified
                            number of consecutive epochs, training will be
                            stopped.
  -b, --batch_size INTEGER  Batch size.
  -l, --lr0 FLOAT           Initial learning rate.
  -m, --momentum FLOAT      Momentum of SGD.
  -o, --out_dir TEXT        Training output directory.
  --batch_size_val INTEGER  Validating batch size. If not set, it will be
                            equal to batch_size.
  --help                    Show this message and exit.
```

### Predict

``````
Usage: predict.py [OPTIONS]

Options:
  -s, --source TEXT      Source of prediction, which can be a path to an
                         image, a video or a directory.  [required]     
  -m, --model-path TEXT  Path to the trained model.  [required]
  -o, --out_dir TEXT     Predicting output directory.
  --help                 Show this message and exit.

``````

