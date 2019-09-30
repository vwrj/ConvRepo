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
batch_size = 512   # input batch size for training
epochs = 20       # number of epochs to train
lr = 0.01      # learning rate

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

class ConvNet(pl.LightningModule):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv_2 = nn.Conv2d(16, 128, 5)
        self.linear_1 = nn.Linear(128*5*5, 64)
        self.linear_2 = nn.Linear(64, 10)

    def forward(self, x):
        # input has shape [bs, 3, 32, 32]
        layer_1 = self.pool(torch.tanh(self.conv_1(x)))
        layer_2 = self.pool(torch.tanh(self.conv_2(layer_1)))
        flatten = layer_2.view(-1, 128*5*5)
        output = self.linear_2(torch.tanh(self.linear_1(flatten)))
        
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

    model = ConvNet()
    # most basic trainer, uses good defaults
    exp = Experiment(save_dir=os.getcwd())
    trainer = Trainer(experiment=exp, gpus=[0, 1], max_nb_epochs=20, distributed_backend='dp') 
    # trainer = Trainer(gpus=[0, 1], max_nb_epochs=20) 
    # trainer = Trainer(experiment=exp, gpus=[0], max_nb_epochs=20) 
    trainer.fit(model) 


