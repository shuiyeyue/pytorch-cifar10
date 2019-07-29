import torch
import torch.nn as nn
import math

class LinearBottleNeck(nn.Module):
    def __init__(self, inps, oups, stride, k, t=6, has_se=False):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(inps, inps * t, kernel_size=1, bias=False),
            nn.BatchNorm2d(inps * t),
            nn.ReLU(inplace=True),
            nn.Conv2d(inps * t, inps * t, kernel_size=k, stride=stride, padding=int((k-1)/2), bias=False, groups=inps * t),
            nn.BatchNorm2d(inps * t),
            nn.ReLU(inplace=True),
            nn.Conv2d(inps * t, oups, kernel_size=1, bias=False),
            nn.BatchNorm2d(oups),
        )
        if has_se:
            self.SE = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(oups, oups // 16, kernel_size=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(oups // 16, oups, kernel_size=1),
                nn.Sigmoid()
            )

        self.stride= stride
        self.inps = inps
        self.oups = oups
        self.has_se = has_se

    def forward(self, x):
        res = self.residual(x)
        if self.has_se:
            res = res * self.SE(res)
        if self.stride == 1 and self.inps == self.oups:
            res += x
        return res

class EfficientNet(nn.Module):
    def __init__(self, has_SE=True, num_classes=100):
        super(EfficientNet, self).__init__()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 3, 6, has_SE)
        self.stage2 = self._make_stage(16, 24, 1, 3, 6, has_SE, 2)
        self.stage3 = self._make_stage(24, 40, 2, 5, 6, has_SE, 2)
        self.stage4 = self._make_stage(40, 80, 2, 3, 6, has_SE, 3)
        self.stage5 = self._make_stage(80, 112,1, 5, 6, has_SE, 3)
        self.stage6 = self._make_stage(112,192,2, 5, 6, has_SE, 4)
        self.stage7 = self._make_stage(192,320,1, 3, 6, has_SE, 1)

        self.last_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classify = nn.Linear(1280, num_classes)
        self._init_weight()

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

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.last_conv(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x



    def _make_stage(self, inps, oups, stride, k, t, has_se, repeat):
        layers = []

        layers += [LinearBottleNeck(inps, oups, stride, k, t, has_se)]
        for _ in range(repeat):
            layers += [LinearBottleNeck(oups, oups, 1, k, t, has_se)]

        return nn.Sequential(*layers)


def efficientnet(**kwargs):
    return EfficientNet()

if __name__ == "__main__":
    model = efficientnet()
    print(model)

    inp = torch.randn(1,3,32,32)
    oup = model(inp)
    print(oup)   



        