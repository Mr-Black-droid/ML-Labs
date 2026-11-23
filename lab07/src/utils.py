import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_fashion_mnist_loaders(batch_size: int = 128, data_dir: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


def count_parameters(model: nn.Module, nonzero_only: bool = False) -> int:
    if not nonzero_only:
        return sum(p.numel() for p in model.parameters())
    nonzero = 0
    for p in model.parameters():
        nonzero += int((p.detach() != 0).sum().item())
    return nonzero


def get_disk_size_mb(model: nn.Module, tmp_path: str = "tmp_ckpt.pt", sparse_format: bool = False) -> float:
    if not sparse_format:
        torch.save(model.state_dict(), tmp_path)
    else:
        sparse_state = {}
        for k, v in model.state_dict().items():
            # Convert weight tensors with zeros to sparse CSR/COO to measure sparse storage
            if v.dim() >= 2 and (v == 0).sum().item() > 0:
                sparse_state[k] = v.to_sparse()
            else:
                sparse_state[k] = v
        torch.save(sparse_state, tmp_path)

    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return size_mb


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def measure_cpu_latency_ms(model: nn.Module, runs: int = 500, warmup: int = 50) -> float:
    """Measures single-sample CPU inference latency (ms) for an honest hardware comparison."""
    cpu_model = model.to("cpu").eval()
    x = torch.randn(1, 1, 28, 28, device="cpu")
    for _ in range(warmup):
        _ = cpu_model(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = cpu_model(x)
    return (time.perf_counter() - t0) * 1000.0 / runs


def train_standard(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int = 6, lr: float = 1e-3):
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()