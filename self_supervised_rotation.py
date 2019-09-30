# Reimplementing the paper: Unsupervised Representation Learning By Predicting Image Rotations

from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_tube import Experiment  
from pytorch_lightning.callbacks import ModelCheckpoint

# options
dataset = 'cifar10' # options: 'mnist' | 'cifar10'
batch_size = 128 # input batch size for training
epochs = 20       # number of epochs to train
lr = 0.1      # learning rate
momentum = 0.9
weight_decay = 5e-4

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = datasets.CIFAR10(root='/scratch/vr1059/cifar10', train=True, download=True, transform=data_transform)
testset = datasets.CIFAR10(root='/scratch/vr1059/cifar10', train=True, download=True, transform=data_transform)

# train_dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
# test_dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# PyTorch Lightning Module

class RotNet(pl.LightningModule):

    def __init__(self):
        super(RotNet, self).__init__()
        
    def forward(self, x):
        # input has shape [bs, 3, 32, 32]

        return output

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.SGD(self.parameters(), lr=lr)

    @pl.data_loader
    def tng_dataloader(self):
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        return test_loader

if __name__ == '__main__':

    model = RotNet()
    # most basic trainer, uses good defaults
    exp = Experiment(save_dir=os.getcwd())
    trainer = Trainer(experiment=exp, gpus=[0, 1], max_nb_epochs=20, distributed_backend='dp') 
    # trainer = Trainer(gpus=[0, 1], max_nb_epochs=20) 
    # trainer = Trainer(experiment=exp, gpus=[0], max_nb_epochs=20) 
    trainer.fit(model) 


