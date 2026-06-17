"""ResNet-18 with GroupNorm (FedCM / Hsieh et al. federated learning setup)."""

import torch.nn as nn


def _gn(channels: int, num_groups: int = 2) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


class BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(planes, num_groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(planes, num_groups)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _gn(planes, num_groups),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18GN(nn.Module):
    """Standard ResNet-18 with BatchNorm replaced by GroupNorm."""

    def __init__(self, num_classes=10, num_groups=2):
        super().__init__()
        self.in_planes = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = _gn(64, num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlockGN(self.in_planes, planes, stride, self.num_groups)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockGN(self.in_planes, planes, 1, self.num_groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
