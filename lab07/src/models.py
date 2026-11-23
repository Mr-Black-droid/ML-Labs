import torch
import torch.nn as nn
import torch.nn.functional as F


class OversizedTeacherCNN(nn.Module):
    """
    ~1.00M parameter baseline CNN for Fashion-MNIST (1x28x28).
    Conv1 (1->64, 3x3)   : 640 params
    Conv2 (64->128, 3x3) : 73,856 params
    FC1 (128*7*7 -> 148) : 928,404 params
    FC2 (148 -> 10)      : 1,490 params
    Total                : ~1,004,774 params
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 148)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(148, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class CompactStudentCNN(nn.Module):
    """
    ~100k parameter student CNN for Fashion-MNIST (1x28x28).
    Conv1 (1->24, 3x3)  : 240 params
    Conv2 (24->48, 3x3) : 10,416 params
    FC1 (48*7*7 -> 38)  : 89,414 params
    FC2 (38 -> 10)      : 390 params
    Total               : ~100,604 params
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 7 * 7, 38)
        self.fc2 = nn.Linear(38, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)