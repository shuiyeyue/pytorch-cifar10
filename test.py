import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cfgs import settings
from utils import get_test_dataloder, get_network

def main():
    model = get_network(args, args.gpu)

    test_datasets = get_test_dataloder(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        args.batch_size,
        args.num_workers
    )

    model.load_state_dict(torch.load(arg.weights), args.gpu)
    print(model)
    
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    for batch_index, (images, labels) in enumerate(test_datasets):
        images = torch.autograd.Variable(images).cuda()
        labels = torch.autograd.Variable(labels).cuda()

        outputs = model(images)

        _, pred = outputs.top(5, 1, largest=True, sorted=True)
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        correct_5 = correct[:, :5].sum()
        correct_1 = correct[:, :1].sum()

        total += len(images)

    print("Top 1 err:", 1 - correct_1 / total)
    print("Top 5 err:", 1 - correct_5 / total)
    print("Param numbers: {}".format(sum(p.numel() for p in mode.parameters())))



if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='model type')
    parser.add_argument('-weight', type=str, required=True, help='model weights')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-num_workers', type=int, default=4, help='workers for dataloader')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size of dataloader')
    
    args = parser.parse_args()

    main()