from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import time
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
batch_size = 512 # input batch size for training
epochs = 20       # number of epochs to train
lr = 0.2      # learning rate

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = datasets.CIFAR10(root='/scratch/vr1059/cifar10', train=True, download=True, transform=data_transform)
testset = datasets.CIFAR10(root='/scratch/vr1059/cifar10', train=True, download=True, transform=data_transform)

# train_dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
# test_dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)

# PyTorch Lightning Module

class ConvNet(pl.LightningModule):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
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
        return torch.optim.SGD(self.parameters(), lr=lr)

    @pl.data_loader
    def tng_dataloader(self):
        train_dist_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_dist_sampler)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True )
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        test_dist_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_dist_sampler)
        # test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return test_loader

def main_process_entrypoint(gpu_nb):
    world = 2
    torch.distributed.init_process_group("nccl", rank=gpu_nb, world_size=world)

    torch.cuda.set_device(gpu_nb)

    model = ConvNet()
    model.cuda(gpu_nb)
    model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[gpu_nb])

    exp = Experiment(save_dir='./logs_cifar10_{}'.format(gpu_nb))
    trainer = Trainer(experiment=exp, gpus=[gpu_nb], max_nb_epochs=20) 
    # trainer = Trainer(gpus=[0, 1], max_nb_epochs=20) 
    # trainer = Trainer(experiment=exp, gpus=[0], max_nb_epochs=20) 
    trainer.fit(model) 



if __name__ == '__main__':

    # torch.multiprocessing.spawn(main_process_entrypoint, nprocs=2)

    model = ConvNet()
    # most basic trainer, uses good defaults
    exp = Experiment(save_dir='./logs_cifar10')
    trainer = Trainer(experiment=exp, gpus=[0, 1], max_nb_epochs=20, distributed_backend='ddp') 
    # trainer = Trainer(experiment=exp, gpus=[0], max_nb_epochs=20) 
    start = time.time()
    trainer.fit(model) 
    print("Amount of time elapsed: {}".format(time.time() - start))


