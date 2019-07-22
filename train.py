import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from cfgs import settings
from utils import get_network, get_test_dataloder, get_training_dataloder, WarmUpLR

def train(model, datasets, optimizer, criterion, epoch, writer, lr_scheduler, warmup_scheduler):
    
    model.train()
    for batch_index, (images, labels) in enumerate(datasets):
        if epoch <= args.warm:
            warmup_scheduler.step()
        
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Traning Epoch: {epoch} [{train_samples}/{total_samples}] \t Loss: {:.4f}\t LR: {:.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
                        epoch=epoch,
            train_samples=batch_index * args.batch_size + len(images),
            total_samples=len(datasets)
            ))

        n_iter = (epoch - 1) * len(datasets) + batch_index + 1
        writer.add_scalar('Train/loss', loss.item(), n_iter)


def eval(model, datasets, criterion, epoch, writer):

    model.eval()
    test_loss = 0.0
    correct = 0.0
    for images, labels in datasets:
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels).cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum()

    print('Test set: Avg Loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(datasets),
        correct / len(datasets)
    ))

    writer.add_scalar('Test/loss', test_loss / len(datasets), epoch)
    writer.add_scalar('Test/Acc', correct / len(datasets), epoch)


def main():
    model = get_network(args.model, args.gpu)
    
    train_datasets = get_training_dataloder(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        args.batch_size,
        args.num_workers
    )
    
    test_datasets = get_test_dataloder(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        args.batch_size,
        args.num_workers
    )

    criterion = nn.CrossEntropyLoss()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='model')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('-lr',type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-warm_up', type=int, default=1, help='warm up train phase')

    args = parser.parse_args()
    
    main()

    



        







        


