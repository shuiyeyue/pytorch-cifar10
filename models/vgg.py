import torch
import torch.nn as nn

import math

__all__ = ['vgg11','vgg13','vgg16','vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn']

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512 ,'M',  512, 512, 512, 512 ,'M']
}

class VGG(nn.Module):
    def __init__(self, cfg, bn=False, num_classes=100):
        super(VGG, self).__init__()
        self.inps = 3
        self.bn = bn

        self.features = self._make_layers(cfg)
        self.classify = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512,num_classes)
        )
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.output_channels
                m.weights.data.normal_(0, math.sqrt(2. / n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weights.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m. nn.BatchNorm2d):
                m.weights.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, cfg):
        layers = []

        for v in cfg:
            if v == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.inps, v, kernel_size=3, stride=1, padding=1, bias=False)]
                
                if self.bn:
                    layers += [nn.BatchNorm2d(v)]
                
                layers += [nn.ReLU(inplace=True)]

                self.inps = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

def vgg11(**kwargs):
    return VGG(cfgs['A'], **kwargs)

def vgg11_bn(**kwargs):
    return VGG(cfgs['A'], bn=True, **kwargs)

def vgg13(**kwargs):
    return VGG(cfgs['B'], **kwargs)

def vgg13_bn(**kwargs):
    return VGG(cfgs['B'], bn=True, **kwargs)

def vgg16(**kwargs):
    return VGG(cfgs['D'], **kwargs)

def vgg16_bn(**kwargs):
    return VGG(cfgs['D'], bn=True, **kwargs)

def vgg19(**kwargs):
    return VGG(cfgs['E'], **kwargs)

def vgg19_bn(**kwargs):
    return VGG(cfgs['E'], bn=True, **kwargs)

if __name__ == "__main__":
    
    model = vgg11()
    print(model)

    input_size = torch.randn(1, 3, 32, 32)
    outputs = model(input_size)
    print(outputs)