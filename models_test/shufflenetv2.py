import torch
import torch.nn as nn

import math

def channel_shuffle(x, g):
    n, c, h, w = x.size()

    x = x.view(n, int(c/g), g, h, w)
    x = x.transpose(1,2).contiguous()
    x = x.view(n, -1, h, w)

    return x

class ShuffleUnit(nn.Module):
    def __init__(self, inps, oups, stride):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        self.inps = inps
        self.oups = oups

        if stride != 1 or self.inps != self.oups:
            self.residual = nn.Sequential(
                nn.Conv2d(inps, inps, kernel_size=1, bias=False),
                nn.BatchNorm2d(inps),
                nn.ReLU(inplace=True),
                nn.Conv2d(inps, inps, kernel_size=3, stride=stride, padding=1, bias=False, groups=inps),
                nn.BatchNorm2d(inps),
                nn.Conv2d(inps, int(oups / 2), kernel_size =1, bias=False),
                nn.BatchNorm2d(int(oups / 2)),
                nn.ReLU(inplace=True),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(inps, inps, kernel_size = 3, stride=stride, padding=1, bias=False, groups=inps),
                nn.BatchNorm2d(inps),
                nn.Conv2d(inps, int(oups/2), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(oups / 2)),
                nn.ReLU(inplace=True),
            )
        else:
            inps = int(self.inps / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(inps, inps, kernel_size=1, bias=False),
                nn.BatchNorm2d(inps),
                nn.ReLU(inplace=True),
                nn.Conv2d(inps, inps, kernel_size=3, stride=stride, padding=1, bias=False, groups=inps),
                nn.BatchNorm2d(inps),
                nn.Conv2d(inps, inps, kernel_size =1, bias=False),
                nn.BatchNorm2d(inps),
                nn.ReLU(inplace=True),
            )

            self.shortcut = nn.Sequential()

    def forward(self, x):
        if self.stride != 1 or self.inps != self.oups:
            x = torch.cat([self.shortcut(x), self.residual(x)], 1)
        else:
            x1 = x[:,0:int(self.inps/2),:,:]
            x2 = x[:,int(self.inps/2): ,:,:]
            x = torch.cat([self.shortcut(x1), self.residual(x2)], 1)

        return channel_shuffle(x, 2)

class ShuffleNetV2(nn.Module):
    def __init__(self, ratio=1.0, num_classes=100):
        super(ShuffleNetV2, self).__init__()

        if ratio == 0.5:
            self.oups = [48, 96, 192, 1024]
        elif ratio == 1.0:
            self.oups = [116, 232, 464, 1024]
        elif ratio == 1.5:
            self.oups = [176, 352, 704, 1024]
        elif ratio == 2.0:
            self.oups = [224, 448, 976, 2048]
        else:
            ValueError('error !')

        
        #self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(24)
        self.relu  = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.stage2 = self._make_layers(24, self.oups[0], 3)
        self.stage3 = self._make_layers(self.oups[0], self.oups[1], 7)
        self.stage4 = self._make_layers(self.oups[1], self.oups[2], 3)

        self.stage5 = nn.Sequential(
            nn.Conv2d(self.oups[2], self.oups[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.oups[3]),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classify = nn.Sequential(
            nn.Linear(self.oups[3], num_classes),
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

    
    def _make_layers(self, inps, oups, repeat):
        layers = []

        layers += [ShuffleUnit(inps, oups, 2)]
        for _ in range(1, repeat):
            layers += [ShuffleUnit(oups, oups, 1)]
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x


def shufflenetv2(**kwargs):
    return ShuffleNetV2(**kwargs)


if __name__ == "__main__":
    model = shufflenetv2(ratio=0.5)
    print(model)

    inp = torch.randn(1,3,224,224)
    oup = model(inp)
    print(oup) 
    print("Param numbers: {}".format(sum(p.numel() for p in model.parameters())))
