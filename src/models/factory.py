"""Build models according to each paper's experimental setup."""

from src.models.cnn import (
    EMNIST_MLP,
    MNIST_L5,
    MNIST_LogisticRegression,
    SimpleCIFAR_CNN,
    SimpleMNIST_CNN,
    WAFFLE_MNIST_L5,
)
from src.models.resnet import ResNet18GN
from src.models.vgg import cifar_vgg16, source_vgg16


def build_model(model_name, num_classes=10, **kwargs):
    name = str(model_name).lower()

    if name in ("mnist_logistic", "mnist_logistic_regression", "emnist_logistic"):
        return MNIST_LogisticRegression(num_classes=num_classes)
    if name in ("emnist_mlp", "emnist_fc"):
        return EMNIST_MLP(num_classes=num_classes)
    if name in ("simple_mnist_cnn", "mnist_cnn"):
        return SimpleMNIST_CNN()
    if name in ("mnist_l5", "waffle_mnist_l5"):
        return MNIST_L5(num_classes=num_classes)
    if name in ("waffle_mnist_l5_strict", "mnist_l5_strict"):
        return WAFFLE_MNIST_L5(num_classes=num_classes)
    if name in ("simple_cifar_cnn", "cifar_cnn"):
        return SimpleCIFAR_CNN()
    if name in ("resnet18_gn", "resnet-18-gn", "resnet18"):
        return ResNet18GN(num_classes=num_classes)
    if name in ("cifar_vgg16", "vgg16", "vgg16_cifar"):
        return cifar_vgg16(num_classes=num_classes)
    if name in ("waffle_cifar_vgg16_strict", "source_vgg16", "waffle_vgg16"):
        return source_vgg16(
            num_classes=num_classes,
            pretrained=bool(kwargs.get("pretrained", False)),
        )

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Supported: mnist_logistic, mnist_l5, emnist_logistic, emnist_mlp, "
        "resnet18_gn, simple_cifar_cnn, cifar_vgg16, waffle_mnist_l5_strict, "
        "waffle_cifar_vgg16_strict"
    )
