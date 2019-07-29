import sys
import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args, use_gpu=True):
  """ return given modelwork """

  if args.model == 'vgg11':
    from models.vgg import vgg11_bn
    model = vgg11_bn()
  elif args.model == 'vgg13':
    from models.vgg import vgg13_bn
    model = vgg13_bn()
  elif args.model == 'vgg16':
    from models.vgg import vgg16_bn
    model = vgg16_bn()
  elif args.model == 'vgg19':
    from models.vgg import vgg19_bn
    model = vgg19_bn()
  elif args.model == 'resnet18':
    from models.resnet import resnet18
    model = resnet18()
  elif args.model == 'resnet50':
    from models.resnet import resnet50
    model = resnet50()
  elif args.model == 'resnet101':
    from models.resnet import resnet101
    model = resnet101()
  elif args.model == 'resnext18':
    from models.resnext import resnext18
    model = resnext18()
  elif args.model == 'resnext50':
    from models.resnext import resnext50
    model = resnext50()
  elif args.model == 'resnext101':
    from models.resnext import resnext101
    model = resnext101()
  elif args.model == 'densenet121':
    from models.densenet import densenet121
    model = densenet121()
  elif args.model == 'densenet169':
    from models.densenet import densenet169
    model = densenet169()
  elif args.model == 'densenet201':
    from models.densenet import densenet201
    model = densenet201()
  elif args.model == 'shufflenetv2':
    from models.shufflenet import shufflenetv2
    model = shufflenetv2()
  elif args.model == 'mobilenetv2':
    from models.mobilenet import mobilnetv2
    model = mobilnetv2()
  elif args.model == 'efficientnet':
    from models.efficientnet import efficientnet
    model = efficientnet()
  else:
    print("modelwork is not supported.")
    sys.exit()
  
  if use_gpu:
    model = model.cuda()

  return model


def get_training_dataloder(mean, std, batch_size=16, num_works=2, shuffle=True):

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
  ])

  cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
  cifar100_training_loader = DataLoader(cifar100_training, batch_size=batch_size, shuffle=True, num_workers=num_works)
  
  return cifar100_training_loader

def get_test_dataloder(mean, std, batch_size=16, num_works=2, shuffle=False):

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
  ])

  cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
  cifar100_test_loader = DataLoader(cifar100_test, shuffle=False, num_workers=num_works ,batch_size=batch_size)

  return cifar100_test_loader


class WarmUpLR(_LRScheduler):
  def __init__(self, optimizier, total_iters, last_epoch=-1):
    self.total_iters = total_iters
    super().__init__(optimizier,last_epoch)

  def get_lr(self):
    return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
