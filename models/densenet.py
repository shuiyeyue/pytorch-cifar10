import torch
import torch.nn as nn

import math

def conv3x3(inps, oups, stride=1):
    return nn.Conv2d(inps, oups, kernel_size=3, padding=1, stride=stride, bias=False)

def conv1x1(inps, oups, stride=1):
    return nn.Conv2d(inps, oups, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, inps, growth_rate):
        super(Bottleneck, self).__init__()

        self.bn1   = nn.BatchNorm2d(inps)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inps, growth_rate * 4)

        self.bn2   = nn.BatchNorm2d(growth_rate * 4)
        self.conv2 = conv3x3(growth_rate * 4, growth_rate)

    def forward(self, x):
        res = x
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = torch.cat([res, x], 1)

        return x

class Transition(nn.Module):
    def __init__(self, inps, oups):
        super(Transition, self).__init__()

        self.bn1   = nn.BatchNorm2d(inps)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inps, oups)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.avg_pool(x)

        return x

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.inps = 2 * growth_rate
        self.conv1 = conv3x3(3, self.inps, stride=2)
        #self.conv1 = conv3x3(3, self.inps, stride=2)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        layers = []
        for index in range(len(nblocks)):
            layers += self._make_layers(block, nblocks[index])
            if index != len(nblocks) - 1:
                self.oups = int(self.inps * reduction)
                layers += [Transition(self.inps, self.oups)]
                self.inps = self.oups
        layers += [
            nn.BatchNorm2d(self.inps),
            nn.ReLU(inplace=True)
        ]
        
        self.features = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classify = nn.Sequential(
            nn.Linear(self.inps, num_classes),
        )

        self._init_weight()


    def forward(self, x):
        x = self.conv1(x)
        #x = self.maxpool(x)
        x = self.features(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def _make_layers(self, block, nblock):
        layers = []
        for _ in range(nblock):
            layers += [block(self.inps, self.growth_rate)]
            self.inps += self.growth_rate
        return layers


def densenet121(**kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169(**kwargs):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201(**kwargs):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161(**kwargs):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

if __name__ == "__main__":
    model = densenet121()
    print(model)

    inp = torch.randn(1,3,32,32)
    oup = model(inp)
    print(oup)


    


