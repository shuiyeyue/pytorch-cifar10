import torch
import torch.nn as nn

import math

class LinearBottleNeck(nn.Module):
    def __init__(self, inps, oups, stride, t=6):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(inps, inps * t, kernel_size=1, bias=False),
            nn.BatchNorm2d(inps * t),
            nn.ReLU(inplace=True),
            nn.Conv2d(inps * t, inps * t, kernel_size=3, stride=stride, padding=1, bias=False, groups=inps * t),
            nn.BatchNorm2d(inps * t),
            nn.ReLU(inplace=True),
            nn.Conv2d(inps * t, oups, kernel_size=1, bias=False),
            nn.BatchNorm2d(oups),
        )

        self.stride= stride
        self.inps = inps
        self.oups = oups

    def forward(self, x):
        res = self.residual(x)

        if self.stride == 1 and self.inps == self.oups:
            res += x
        
        return res

class MobileNetV2(nn.Module):
    def __init__(self, t=6, num_classes=100):
        super(MobileNetV2, self).__init__()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.stage1 = LinearBottleNeck(32 , 16,  1, 1)
        self.stage2 = self._make_stage(16 , 24,  2, 6, 2)
        self.stage3 = self._make_stage(24 , 32,  1, 6, 3)
        self.stage4 = self._make_stage(32 , 64,  2, 6, 4)
        self.stage5 = self._make_stage(64 , 96,  1, 6, 3)
        self.stage6 = self._make_stage(96 , 160, 2, 6, 3)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv_last = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classify = nn.Conv2d(1280, num_classes, kernel_size=1)

        self._init_weight()

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x =self.conv_last(x)

        x = self.avg_pool(x)
        x = self.classify(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_stage(self, inps, oups, stride, t, repeat):
        layers = []

        layers += [LinearBottleNeck(inps, oups, stride=stride, t=t)]
        for _ in range(1, repeat):
            layers += [LinearBottleNeck(oups, oups, 1, t)]

        return nn.Sequential(*layers)
    
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


def mobilnetv2(**kwargs):
    return MobileNetV2()

if __name__ == "__main__":
    model = mobilnetv2()
    print(model)

    inp = torch.randn(1,3,32,32)
    oup = model(inp)
    print(oup)   
