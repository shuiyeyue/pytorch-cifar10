import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math

__all__ = ['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn']

class VGG(nn.Module):
  def __init__(self, features, num_classes=10):
    super(VGG, self).__init__()
    self.features = features
    self.classify = nn.Sequential(
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(512,num_classes),
      nn.ReLU(inplace=True),
    )

    self._init_weight()

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2./n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
  
  def forward(self, x):
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classify(out)

    return out


def make_layers(config, batch_norm=False):
  in_channels = 3
  layers = []
  for v in config:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      if batch_norm:
        layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)]    
      in_channels = v

  return nn.Sequential(*layers)


cfgs = {
  'A': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
  'B': [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
  'D': [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
  'E': [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

def vgg11(**kwargs):
  return VGG(make_layers(cfgs['A']),**kwargs)

def vgg11_bn(**kwargs):
  return VGG(make_layers(cfgs['A'],batch_norm=True),**kwargs)

def vgg13(**kwargs):
  return VGG(make_layers(cfgs['B']),**kwargs)

def vgg13_bn(**kwargs):
  return VGG(make_layers(cfgs['B'],batch_norm=True),**kwargs)

def vgg16(**kwargs):
  return VGG(make_layers(cfgs['D']),**kwargs)

def vgg16_bn(**kwargs):
  return VGG(make_layers(cfgs['D'],batch_norm=True),**kwargs)

def vgg19(**kwargs):
  return VGG(make_layers(cfgs['E']),**kwargs)

def vgg19_bn(**kwargs):
  return VGG(make_layers(cfgs['E'],batch_norm=True),**kwargs)


if __name__ == "__main__":
  net = vgg19_bn()
  print(net)

  input_size = (1,3,32,32)
  x = torch.randn(input_size)
  out = net(x)
  print(out)
