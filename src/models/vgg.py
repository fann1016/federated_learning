import torch
import torch.nn as nn
from torchvision import models
import os


_VGG_CONFIGS = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


def _make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class CIFARVGG16(nn.Module):
    """VGG-16 classifier for CIFAR-sized 32x32 images."""

    def __init__(self, num_classes=10, batch_norm=False, dropout=0.5):
        super().__init__()
        self.features = _make_layers(_VGG_CONFIGS["D"], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def cifar_vgg16(num_classes=10):
    return CIFARVGG16(num_classes=num_classes, batch_norm=False)


def source_vgg16(num_classes=10, pretrained=False):
    """Torchvision VGG16 path used by the WAFFLE source for CIFAR-10."""
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        torch_home = os.path.abspath(os.path.join(os.getcwd(), ".torch"))
        os.environ["TORCH_HOME"] = torch_home
    torch.hub.set_dir(torch_home)

    if pretrained:
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=weights)
        except AttributeError:
            model = models.vgg16(pretrained=True)
    else:
        try:
            model = models.vgg16(weights=None)
        except TypeError:
            model = models.vgg16(pretrained=False)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model
