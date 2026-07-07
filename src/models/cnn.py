import torch.nn as nn
import torch.nn.functional as F

class SimpleCIFAR_CNN(nn.Module):
    def __init__(self):
        super(SimpleCIFAR_CNN, self).__init__()
        # Convolutional layers: 64, 64, 128, 128 (McMahan et al. 2017)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers: 384, 192, 10
        self.fc1 = nn.Linear(128 * 8 * 8, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MNIST_LogisticRegression(nn.Module):
    """Multinomial logistic regression for FedProx MNIST experiments (784 -> 10)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MNIST_L5(nn.Module):
    """5-layer MNIST CNN used by the WAFFLE experiments."""

    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.block(x)
        x = x.view(-1, 128 * 5 * 5)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class WAFFLE_MNIST_L5(MNIST_L5):
    """WAFFLE source-compatible MNIST L5: log-probabilities for NLL loss."""

    def forward(self, x):
        return F.log_softmax(super().forward(x), dim=1)


class EMNIST_MLP(nn.Module):
    """2-layer fully connected network (SCAFFOLD paper Table 5, non-convex)."""

    def __init__(self, num_classes=62, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleMNIST_CNN(nn.Module):
    def __init__(self):
        super(SimpleMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
