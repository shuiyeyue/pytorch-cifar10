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

def train(model, datasets, optimizer, criterion, epoch, writer, warmup_scheduler):
    
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
            loss.item() / len(images),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            train_samples=batch_index + 1,
            total_samples=len(datasets)
            ))

        n_iter = (epoch - 1) * len(datasets) + batch_index + 1
        writer.add_scalar('Train/loss', loss.item() / len(images), n_iter)


def eval(model, datasets, criterion, epoch, writer):

    model.eval()
    test_loss = 0.0
    correct = 0.0
    num_data = 0.0
    for images, labels in datasets:
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels).cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum()
        num_data += len(images)

    print('Test set: Avg Loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / num_data,
        correct / num_data
    ))

    writer.add_scalar('Test/loss', test_loss / num_data, epoch)
    writer.add_scalar('Test/Acc', correct / num_data, epoch)

    return correct.float() / num_data


def main():
    model = get_network(args, args.gpu)
    
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
    optimizier = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_sheduler = optim.lr_scheduler.MultiStepLR(optimizier, milestones=settings.MILESTONES, gamma=0.1, last_epoch=-1)
    iter_per_epoch = len(train_datasets)
    warmup_scheduler = WarmUpLR(optimizier, iter_per_epoch * args.warm_up)
    checkpoints_path = os.path.join(settings.CHECKPOINT_PATH, args.model, settings.TIME_NOW)

    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.model, settings.TIME_NOW))

    #input_tensor = torch.Tensor(args.batch_size, 3, 32, 32).cuda()
    #writer.add_graph(model, torch.autograd.Variable(input_tensor, required_grad=True))

    if not os.path.join(checkpoints_path):
        os.makedirs(checkpoints_path)
    checkpoints_path = os.path.join(checkpoints_path, '{model}_{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm_up:
            train_sheduler.step(epoch)
        
        train(model, train_datasets, optimizier, criterion, epoch, writer, warmup_scheduler)
        acc = eval(model, test_datasets, criterion, epoch, writer)

        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(model.state_dict(), checkpoints_path.format(model=args.model, epoch=epoch, type='best'))
            best_acc = acc
            
        if not epoch % settings.SAVE_EPOCH:
            torch.save(model.state_dict(), checkpoints_path.format(model=args.model, epoch=epoch, type='regular'))

    writer.close()

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