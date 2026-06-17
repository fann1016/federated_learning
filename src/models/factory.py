"""Build models according to each paper's experimental setup."""

from src.models.cnn import (
    EMNIST_MLP,
    MNIST_LogisticRegression,
    SimpleCIFAR_CNN,
    SimpleMNIST_CNN,
)
from src.models.resnet import ResNet18GN


def build_model(model_name, num_classes=10):
    name = str(model_name).lower()

    if name in ("mnist_logistic", "mnist_logistic_regression", "emnist_logistic"):
        return MNIST_LogisticRegression(num_classes=num_classes)
    if name in ("emnist_mlp", "emnist_fc"):
        return EMNIST_MLP(num_classes=num_classes)
    if name in ("simple_mnist_cnn", "mnist_cnn"):
        return SimpleMNIST_CNN()
    if name in ("simple_cifar_cnn", "cifar_cnn"):
        return SimpleCIFAR_CNN()
    if name in ("resnet18_gn", "resnet-18-gn", "resnet18"):
        return ResNet18GN(num_classes=num_classes)

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Supported: mnist_logistic, emnist_logistic, emnist_mlp, resnet18_gn, simple_cifar_cnn"
    )
