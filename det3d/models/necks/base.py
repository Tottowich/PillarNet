import spconv
from torch import nn
from ..utils import build_norm_layer

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, minplanes, dilation=1, norm_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, minplanes, kernel_size=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, minplanes)[1]
        self.conv2 = nn.Conv2d(minplanes, minplanes, kernel_size=3, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, minplanes)[1]
        self.conv3 = nn.Conv2d(minplanes, inplanes, kernel_size=1, bias=False)
        self.bn3 = build_norm_layer(norm_cfg, inplanes)[1]
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class Bottlenecka(nn.Module):
    expansion = 2

    def __init__(self, inplanes, minplanes, dilation=1, norm_cfg=None):
        super(Bottlenecka, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, minplanes, kernel_size=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, minplanes)[1]
        self.conv2 = nn.Conv2d(minplanes, minplanes, kernel_size=3, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, minplanes)[1]
        self.conv3 = nn.Conv2d(minplanes, inplanes, kernel_size=1, bias=False)
        self.bn3 = build_norm_layer(norm_cfg, inplanes)[1]
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = out + residual
        return out

class BottleneckA(nn.Module):
    expansion = 2

    def __init__(self, inplanes, stride=1, downsample=None, norm_cfg=None):
        super(BottleneckA, self).__init__()
        planes = inplanes // BottleneckA.expansion
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = nn.Conv2d(planes, planes * BottleneckA.expansion, kernel_size=1, bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * BottleneckA.expansion)[1]
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckB(nn.Module):
    expansion = 2

    def __init__(self, inplanes, stride=1, downsample=None, norm_cfg=None):
        super(BottleneckB, self).__init__()
        planes = inplanes // BottleneckA.expansion
        assert stride == 1, 'stride != 1'
        assert downsample is None, 'downsample is not None'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)  # stride = 1, dilation = 2
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = nn.Conv2d(planes, planes * BottleneckB.expansion, kernel_size=1, bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * BottleneckB.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.extra_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * BottleneckB.expansion, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, planes * BottleneckB.expansion)[1]
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.extra_conv(x)

        if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out