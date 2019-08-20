import torch
import torch.nn as nn
import torch.nn.functional as F

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            #Hsigmoid()
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LinearBottleNeck(nn.Module):
    def __init__(self, inps, oups, stride, k, t=6, has_se=False):
        super(LinearBottleNeck, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(inps, inps * t, kernel_size=1, bias=False),
            nn.BatchNorm2d(inps * t),
            Hswish(inplace=True),
            nn.Conv2d(inps * t, inps * t, kernel_size=k, stride=stride, padding=int((k-1)/2), bias=False, groups=inps * t),
            nn.BatchNorm2d(inps * t),
            Hswish(inplace=True),
            nn.Conv2d(inps * t, oups, kernel_size=1, bias=False),
            nn.BatchNorm2d(oups),
        )
        if has_se:
            self.residual = nn.Sequential(
                nn.Conv2d(inps, inps * t, kernel_size=1, bias=False),
                nn.BatchNorm2d(inps * t),
                Hswish(inplace=True),
                nn.Conv2d(inps * t, inps * t, kernel_size=k, stride=stride, padding=int((k-1)/2), bias=False, groups=inps * t),
                nn.BatchNorm2d(inps * t),
                Hswish(inplace=True),
                SEModule(inps * t),
                nn.Conv2d(inps * t, oups, kernel_size=1, bias=False),
                nn.BatchNorm2d(oups),
            )

        self.stride= stride
        self.inps = inps
        self.oups = oups
        self.has_se = has_se

    def forward(self, x):
        res = self.residual(x)
        if self.stride == 1 and self.inps == self.oups:
            res += x
        return res

class MBConv(nn.Module):
    def __init__(self, inplanes, planes, repeat, kernel_size, stride, expand):
        super(MBConv, self).__init__()
        layer = []

        layer.append(LinearBottleNeck(inplanes, planes, stride, kernel_size, expand, True))

        for i in range(1, repeat):
            layer.append(LinearBottleNeck(planes, planes, 1, kernel_size, expand, True))

        self.layer = nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.layer(x)
        return out

class Upsample(nn.Module):
    def __init__(self, scale):
        super(Upsample, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

def make_divisible(v, divisior=8, min_value=None):
    if min_value is None:
        min_value = divisior

    new_v = max(min_value, int(v + divisior / 2) // divisior * divisior)
    if new_v < 0.9 * v:
        new_v += divisior
    
    return new_v


class EfficientNet(nn.Module):
    def __init__(self, num_classes=100, width_coef=1., depth_coef=1., scale=1.):
        super(EfficientNet,self).__init__()

        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        expands  = [1, 6, 6, 6, 6, 6, 6]
        repeats  = [1, 2, 2, 3, 3, 4, 1]
        #strides  = [1, 2, 2, 2, 1, 2, 1]
        strides  = [1, 1, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [make_divisible(x * width) for x in channels]
        repeats  = [round(x * depth) for x in repeats]
        self.upsample = Upsample(scale)

        self.stage1 = nn.Sequential(
            #nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0])
        )
        
        self.stage2 = MBConv(channels[0], channels[1], repeats[0], kernel_size[0], strides[0], expands[0])
        self.stage3 = MBConv(channels[1], channels[2], repeats[1], kernel_size[1], strides[1], expands[1])
        self.stage4 = MBConv(channels[2], channels[3], repeats[2], kernel_size[2], strides[2], expands[2])
        self.stage5 = MBConv(channels[3], channels[4], repeats[3], kernel_size[3], strides[3], expands[3])
        self.stage6 = MBConv(channels[4], channels[5], repeats[4], kernel_size[4], strides[4], expands[4])
        self.stage7 = MBConv(channels[5], channels[6], repeats[5], kernel_size[5], strides[5], expands[5])
        self.stage8 = MBConv(channels[6], channels[7], repeats[6], kernel_size[6], strides[6], expands[6])

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Hswish(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classify = nn.Sequential(
           nn.Dropout(),
           nn.Linear(channels[8], num_classes), 
        )

        self._init_params()
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

def efficientnet_b0(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0)

def efficientnet_b1(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224.)

def efficientnet_b2(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224.)

def efficientnet_b3(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224.)

def efficientnet_b4(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224.)

def efficientnet_b5(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224.)

def efficientnet_b6(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224.)

def efficientnet_b7(num_classes=100):
    return EfficientNet(num_classes=num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224.)

def test():
    x = torch.FloatTensor(1, 3, 224, 224)
    model = efficientnet_b0(num_classes=100)
    print(model)
    out = model(x)
    print(out.size())
    print("Param numbers: {}".format(sum(p.numel() for p in model.parameters()))) 

if __name__ == '__main__':
    test()


