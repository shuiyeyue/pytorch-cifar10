import torch
import torch.nn as nn

import math

def conv3x3(inps, oups, stride=1, groups=1):
    return nn.Conv2d(inps, oups, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)

def conv1x1(inps, oups, stride=1):
    return nn.Conv2d(inps, oups, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 2
    def __init__(self, inps, oups, stride=1, groups=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inps, oups * 2)
        self.bn1   = nn.BatchNorm2d(oups * 2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(oups * 2, oups * 2, stride=stride, groups=groups)
        self.bn2   = nn.BatchNorm2d(oups * 2)

        self.downsample = downsample

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if not self.downsample is None:
            res = self.downsample(res)

        res += x
        x = self.relu(x)

        return x

class BottleBlock(nn.Module):
    expansion = 4
    def __init__(self, inps, exps, stride=1, groups=1, downsample=None):
        super(BottleBlock, self).__init__()

        self.conv1 = conv1x1(inps, exps * 2)
        self.bn1   = nn.BatchNorm2d(exps * 2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(exps * 2, exps * 2, stride=stride, groups=groups)
        self.bn2   = nn.BatchNorm2d(exps * 2)
        self.conv3 = conv1x1(exps * 2, exps * self.expansion)
        self.bn3   = nn.BatchNorm2d(exps * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if not self.downsample is None:
            res = self.downsample(res)

        x += res
        x = self.relu(x)

        return x

class ResNext(nn.Module):
    def __init__(self, block, num_layers, groups=32, num_classes=100):
        super(ResNext, self).__init__()
        self.inps = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()

        self.layer1 = self._make_layers(block, 64 , num_layers[0], 2, groups)
        self.layer2 = self._make_layers(block, 128, num_layers[1], 1, groups)
        self.layer3 = self._make_layers(block, 256, num_layers[2], 2, groups)
        self.layer4 = self._make_layers(block, 512, num_layers[3], 2, groups)

        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weights.data.normal_(0, math.sqrt(2./n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weights.data.normal_(0, 0.01)
                m.bias.data.zero_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weights.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, oups, num_layer, stride, groups):
        layers = []

        downsample = None
        if stride != 1 or self.inps != oups * block.expansion: 
            downsample = nn.Sequential(
                conv1x1(self.inps, oups * block.expansion, stride=stride),
                nn.BatchNorm2d(oups * block.expansion),
            )

        layers += [block(self.inps, oups, stride=stride, groups=groups, downsample=downsample)]
        self.inps = oups * block.expansion

        for _ in range(1, num_layer):
            layers += [block(self.inps, oups, stride=1, groups=groups)]

        return nn.Sequential(*layers)

def resnext18(**kwargs):
    return ResNext(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnext36(**kwargs):
    return ResNext(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnext50(**kwargs):
    return ResNext(BottleBlock,[3, 4, 6, 3], **kwargs)

def resnext101(**kwargs):
    return ResNext(BottleBlock,[3, 4, 23, 3], **kwargs)

def resnext152(**kwargs):
    return ResNext(BottleBlock,[3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    model = resnext50()
    print(model)

    inp = torch.randn(1,3,32,32)
    oup = model(inp)
    print(oup)      
