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

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_tube import Experiment  
from pytorch_lightning.callbacks import ModelCheckpoint

# options
dataset = 'cifar10' # options: 'mnist' | 'cifar10'
batch_size = 128 # input batch size for training
epochs = 100       # number of epochs to train
lr = 0.1      # learning rate
momentum = 0.9
weight_decay = 5e-4

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = datasets.CIFAR10(root='/scratch/vr1059/cifar10', train=True, download=True, transform=data_transform)
testset = datasets.CIFAR10(root='/scratch/vr1059/cifar10', train=True, download=True, transform=data_transform)


def rotate_90(img, n):
    return np.rot90(img, n, (1, 2))

def collate(batch):
    imgs = []
    labels = []
    for (x, y) in batch:
        # TODO: we can try randomizing the rotation here. 
        for n in [0, 1, 2, 3]:
            imgs.append(torch.FloatTensor(rotate_90(x.numpy(), n).copy()))
            labels.append(torch.tensor(n))

    return [torch.stack(imgs), torch.stack(labels)]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(BasicBlock, self).__init__()

        padding = (int)((filter_size-1)/2)
        self.layers = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, stride=1, padding=padding, bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
            ]))

    def forward(self, x):
        return self.layers(x)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)


class NetworkInNetwork(pl.LightningModule):
    def __init__(self):
        super(NetworkInNetwork, self).__init__()

        
        
# PyTorch Lightning Module

class RotNet(pl.LightningModule):

    def __init__(self):
        super(RotNet, self).__init__()

        # Hard-coded to run first experiment, will make modular afterwards. 

        num_classes = 4
        num_blocks = 3
                        
        n_channels = 192
        blocks = []

        blocks.append(nn.Sequential(OrderedDict([
            ('b1_convb1', BasicBlock(3, 192, 5)),
            ('b1_convb2', BasicBlock(192, 160, 1)),
            ('b1_convb3', BasicBlock(160, 96, 1)),
            ('b1_maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ])))

        blocks.append(nn.Sequential(OrderedDict([
            ('b2_convb1', BasicBlock(96, n_channels, 5)),
            ('b2_convb2', BasicBlock(n_channels, n_channels, 1)),
            ('b2_convb3', BasicBlock(n_channels, n_channels, 1)),
            ('b2_avgpool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            ])))

        blocks.append(nn.Sequential(OrderedDict([
            ('b3_convb1', BasicBlock(n_channels, n_channels, 3)),
            ('b3_convb2', BasicBlock(n_channels, n_channels, 1)),
            ('b3_convb3', BasicBlock(n_channels, n_channels, 1)),
            ('global_avgpool', GlobalAveragePooling()),
            ('classifier', nn.Linear(n_channels, num_classes)),
            ])))

        self.blocks = nn.ModuleList(blocks) 

        
    def forward(self, x):
        # input has shape [bs, 3, 32, 32]
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        pred = y_hat.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(y.data.view_as(pred)).sum()
        return {'val_loss': F.cross_entropy(y_hat, y), 'correct': correct, 'bs': y.shape[0]}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_set_size = torch.tensor([x['bs'] for x in outputs]).sum()
        avg_accuracy = torch.stack([x['correct'] for x in outputs]).sum().item()/(1. * val_set_size.item())
        return {'avg_val_loss': avg_loss, 'avg_val_acc': 100*avg_accuracy}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    @pl.data_loader
    def tng_dataloader(self):
        train_dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, collate_fn=collate, batch_size=batch_size, shuffle=False, sampler=train_dist_sampler)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        test_dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        test_loader  = torch.utils.data.DataLoader(testset, collate_fn=collate, batch_size=batch_size, shuffle=False, sampler=test_dist_sampler)
        return test_loader

if __name__ == '__main__':

    model = RotNet()
    # most basic trainer, uses good defaults
    exp = Experiment(save_dir='./logs_rotnet')
    trainer = Trainer(experiment=exp, gpus=[0, 1], max_nb_epochs=20, distributed_backend='ddp') 
    # trainer = Trainer(gpus=[0, 1], max_nb_epochs=20) 
    # trainer = Trainer(experiment=exp, gpus=[0], max_nb_epochs=20) 
    trainer.fit(model) 


