import torch
import torch.nn as nn
import math

def conv3x3(inps, oups, stride=1):
    return nn.Conv2d(inps, oups, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(inps, oups, stride=1):
    return nn.Conv2d(inps, oups, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inps, oups, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inps, oups)
        self.bn1   = nn.BatchNorm2d(oups)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(oups, oups, stride=stride)
        self.bn2    = nn.BatchNorm2d(oups)

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

        x += res
        x = self.relu(x)

        return x

class BottleBlock(nn.Module):
    expansion = 4
    def __init__(self, inps, exps, stride=1, downsample=None):
        super(BottleBlock, self).__init__()

        self.conv1 = conv1x1(inps, exps)
        self.bn1   = nn.BatchNorm2d(exps)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(exps, exps, stride=stride)
        self.bn2   = nn.BatchNorm2d(exps)
        self.conv3 = conv1x1(exps, exps * self.expansion)
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

class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes=100):
        super(ResNet, self).__init__()
        self.inps = 64

        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layers(block, num_layers[0], 64,  2)
        self.layer2 = self._make_layers(block, num_layers[1], 128, 1)
        self.layer3 = self._make_layers(block, num_layers[2], 256, 2)
        self.layer4 = self._make_layers(block, num_layers[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classify = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 2./math.sqrt(n))
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
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

    def _make_layers(self, block, num_layer, num_channel, stride):
        layers = []

        downsample = None
        if stride != 1 or  self.inps != num_channel * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inps, num_channel * block.expansion, stride=stride),
                nn.BatchNorm2d(num_channel * block.expansion),
            )

        layers += [block(self.inps, num_channel, stride, downsample)]
        self.inps = num_channel * block.expansion
        for i in range(1, num_layer):
            layers += [block(self.inps, num_channel)]

        return nn.Sequential(*layers)
        
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet36(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(BottleBlock,[3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(BottleBlock,[3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(BottleBlock,[3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    model = resnet50()
    print(model)

    inp = torch.randn(1,3,32,32)
    oup = model(inp)
    print(oup)        