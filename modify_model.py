# coding: utf-8

import torch
from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter

model_path = './output/1000_0.01_dropout0.7/epoch_17.pd'
log_path = './log'


def load_model(path):
    return torch.load(path)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = load_model(model_path)

    p = (
        (model.conv1[0], 'weight'),
        (model.conv2[0], 'weight'),
        (model.fc[0], 'weight'),
        (model.fc[3], 'weight'),
    )
    prune.global_unstructured(
        p,
        prune.L1Unstructured,
        amount=0.2
    )

    with SummaryWriter(log_dir=log_path) as writer:
        writer.add_graph(model, (torch.rand(2, 28, 28),))
