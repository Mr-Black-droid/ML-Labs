import os
import random
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from thop import profile

# ==========================================
# 0. Global Seeds & Version Logging
# ==========================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.version}")


# ==========================================
# 1. Dataset Loading & Preprocessing
# ==========================================
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_val_dataset = torchvision.datasets.FashionMNIST(
        root='data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='data', train=False, download=True, transform=transform
    )

    # Split official train 60k into 55k train and 5k validation
    indices = torch.randperm(len(train_val_dataset), generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_indices = indices[:55000]
    val_indices = indices[55000:60000]

    train_dataset = Subset(train_val_dataset, train_indices)
    val_dataset = Subset(train_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader


# ==========================================
# 2. Model Definitions (Target 100k-150k params)
# ==========================================
class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2 -> Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2 -> Flatten -> Linear(32*7*7, 10)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


class FashionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Split 28x28 into 16 patches of 7x7. Patch dim = 1 * 7 * 7 = 49.
        # Linear projection to 64. Two Transformer encoder layers: d_model=64, nhead=4, dim_feedforward=128. Mean pool, Linear 64->10
        self.patch_dim = 49
        self.proj = nn.Linear(self.patch_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape into patches: (B, C, 4 patches in H, 7, 4 patches in W, 7) -> (B, 16, 49)
        x = x.unfold(2, 7, 7).unfold(3, 7, 7) # (B, 1, 4, 4, 7, 7)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, 16, -1) # (B, 16, 49)
        
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Mean pooling over sequence dimension
        return self.fc(x)


# ==========================================
# 3. Cost & Latency Benchmarking
# ==========================================
def benchmark_models(device):
    models = {
        "MLP": FashionMLP().to(device),
        "CNN": FashionCNN().to(device),
        "Attention": FashionAttention().to(device)
    }
    
    results = {}
    dummy_input_b1 = torch.randn(1, 1, 28, 28).to(device)
    dummy_input_b128 = torch.randn(128, 1, 28, 28).to(device)

    print("\n--- Model Cost & Latency Analysis ---")
    for name, model in models.items():
        # Parameter count
        param_summary = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        total_params = param_summary.total_params

        # FLOPs via thop
        macs, _ = profile(model, inputs=(dummy_input_b1,), verbose=False)
        flops = macs * 2

        # Latency benchmarking (Warm-up + Median of 100 for batch 1, 20 for batch 128)
        model.eval()
        with torch.no_grad():
            # Warm-up batch 1
            for _ in range(5):
                _ = model(dummy_input_b1)
            lat_b1_runs = []
            for _ in range(100):
                t0 = time.perf_counter()
                _ = model(dummy_input_b1)
                t1 = time.perf_counter()
                lat_b1_runs.append((t1 - t0) * 1000)
            lat_b1 = np.median(lat_b1_runs)

            # Warm-up batch 128
            for _ in range(5):
                _ = model(dummy_input_b128)
            lat_b128_runs = []
            for _ in range(20):
                t0 = time.perf_counter()
                _ = model(dummy_input_b128)
                t1 = time.perf_counter()
                lat_b128_runs.append((t1 - t0) * 1000)
            lat_b128 = np.median(lat_b128_runs)

        results[name] = {
            "params": total_params,
            "flops": flops,
            "lat_b1": lat_b1,
            "lat_b128": lat_b128
        }
        print(f"[{name}] Params: {total_params:,} | FLOPs: {flops:,} | Latency (B=1): {lat_b1:.2f}ms | Latency (B=128): {lat_b128:.2f}ms")

    return models, results


# ==========================================
# 4. Training Loop (5 epochs, Adam, lr=1e-3)
# ==========================================
def train_and_evaluate(models, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    accuracies = {}

    print("\n--- Training Models (5 Epochs) ---")
    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(5):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate test accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100.0 * correct / total
        accuracies[name] = acc
        print(f"{name} Test Accuracy: {acc:.2f}%")

    return accuracies


# ==========================================
# 5. Plotting Trade-offs
# ==========================================
def generate_plots(results, accuracies):
    os.makedirs("lab03/results", exist_ok=True)
    names = list(results.keys())
    params = [results[n]["params"] for n in names]
    lat_b1 = [results[n]["lat_b1"] for n in names]
    accs = [accuracies[n] for n in names]

    # Plot Accuracy vs Parameters
    plt.figure(figsize=(8, 5))
    plt.scatter(params, accs, color=['blue', 'green', 'red'], s=100)
    for i, txt in enumerate(names):
        plt.annotate(txt, (params[i], accs[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Parameter Count")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Parameter Count")
    plt.grid(True)
    plt.savefig("lab03/results/accuracy_vs_params.png")
    plt.close()

    # Plot Accuracy vs Latency (Batch 1)
    plt.figure(figsize=(8, 5))
    plt.scatter(lat_b1, accs, color=['blue', 'green', 'red'], s=100)
    for i, txt in enumerate(names):
        plt.annotate(txt, (lat_b1[i], accs[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Inference Latency Batch=1 (ms)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Latency (Batch Size = 1)")
    plt.grid(True)
    plt.savefig("lab03/results/accuracy_vs_latency.png")
    plt.close()
    print("\nSuccessfully generated and saved trade-off plots in lab03/results/")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()
    models, results = benchmark_models(device)
    accuracies = train_and_evaluate(models, train_loader, test_loader, device)
    generate_plots(results, accuracies)