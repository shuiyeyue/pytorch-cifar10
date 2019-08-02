import torch
import torch.nn as nn
import torch.nn.init as init

from collections import OrderedDict

def _make_divisible(v, divisior, min_value=None):
    if min_value is None:
        min_value = divisior

    new_v = max(min_value, int(v + divisior / 2) // divisior * divisior)
    if new_v < 0.9 * v:
        new_v += divisior
    
    return new_v

class LinearBottleneck(nn.Module):
    def __init__(self, inps, oups, stride, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inps, inps * t, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(inps * t)
        self.conv2 = nn.Conv2d(inps * t, inps * t, kernel_size=3, stride=stride, padding=1, bias=False, groups=inps * t)
        self.bn2   = nn.BatchNorm2d(inps * t)
        self.conv3 = nn.Conv2d(inps * t, oups, kernel_size=1, stride=1, bias=False)
        self.bn3   = nn.BatchNorm2d(oups)
        
        self.inps = inps
        self.oups = oups
        self.stride = stride
        self.activation = activation(inplace=True)

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.inps == self.oups and self.stride == 1:
            x += res

        return x

class MobilenetV2(nn.Module):
    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=100, activation=nn.ReLU6):
        super(MobilenetV2, self).__init__()

        self.scale = scale
        self.t = t
        self.activation = activation(inplace=True)
        self.activation_type = activation
        self.num_classes = num_classes

        self.num_channels = [32, 16 ,24, 32, 64, 96, 160 ,320]
        assert input_size % 32 == 0 

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.c[0], kernel_size=3, stride=self.s[0], padding=1, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU6(inplace=True),
        )
        self.bottlenecks = self._make_bottlenecks()

        self.last_out_channels = 1280 if self.scale <=1.0 else _make_divisible(1280 * self.scale, 8)
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.c[-1], self.last_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_out_channels),
            nn.ReLU6()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.classify = nn.Linear(self.last_out_channels, num_classes)
        
        self.init_params()

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        bottleneck1 = LinearBottleneck(self.c[0], self.c[1], self.s[1], self.t, self.activation_type)
        modules[stage_name + '_0'] = bottleneck1

        for i in range(1, len(self.c) - 1):
            name = stage_name + '_{}'.format(i)
            modules[name] = self._make_stage(self.c[i], self.c[i+1], self.s[i+1], self.t, self.n[i+1], stage=i)

        return nn.Sequential(modules)

    def _make_stage(self, inps, oups, stride, t, repeat, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        first_module = LinearBottleneck(inps, oups, stride, t, self.activation_type)
        modules[stage_name + '_0'] = first_module
        for i in range(1, repeat):
            name = stage_name + "_{}".format(i)
            modules[name] = LinearBottleneck(oups, oups, 1, 6, self.activation_type)

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.bottlenecks(x)
        x = self.last_conv(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        
        return x
    
    def init_params(self):
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


if __name__ == "__main__":
    model = MobilenetV2(scale=1.0, num_classes=100)
    print(model)

    inp = torch.randn(1,3,224,224)
    oup = model(inp)
    print(oup)
    print("Param numbers: {}".format(sum(p.numel() for p in model.parameters())))


