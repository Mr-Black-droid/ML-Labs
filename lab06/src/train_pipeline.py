import os
import time
import copy
import yaml
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity

# ==============================================================================
# Reproducibility & Model Architecture (Compact CNN from Lab 3)
# ==============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_config_paths(cfg, config_dir):
    for section, key in (("dataset", "data_dir"), ("checkpointing", "save_dir"), ("checkpointing", "resume_from")):
        value = cfg[section].get(key)
        if value and not os.path.isabs(value):
            cfg[section][key] = os.path.abspath(os.path.join(config_dir, value))

class CompactCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_dataloaders(cfg, batch_size=None, num_workers=None):
    bs = batch_size if batch_size is not None else cfg["training"]["batch_size"]
    nw = num_workers if num_workers is not None else cfg["training"]["num_workers"]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train = datasets.FashionMNIST(cfg["dataset"]["data_dir"], train=True, download=True, transform=transform)
    val_len = int(len(full_train) * cfg["dataset"]["val_split"])
    train_len = len(full_train) - val_len
    
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=generator)
    
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin_mem)
    return train_loader, val_loader

# ==============================================================================
# Task 1 & 7: YAML-Driven MLflow Training & Checkpoint Resumption
# ==============================================================================
def build_optimizer_and_scheduler(model, cfg):
    opt_name = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr * 10, momentum=cfg["training"]["momentum"], weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    sched_name = cfg["training"]["scheduler"].lower()
    if sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["training"]["step_size"], gamma=cfg["training"]["gamma"])
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    elif sched_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    else:
        scheduler = None

    return optimizer, scheduler

def train_pipeline(cfg, max_epochs_override=None, run_name_override=None):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpointing"]["save_dir"], exist_ok=True)
    
    mlflow.set_experiment(cfg["experiment_name"])
    run_name = run_name_override or cfg["run_name"]
    
    train_loader, val_loader = get_dataloaders(cfg)
    model = CompactCNN(cfg["model"]["num_classes"], cfg["model"]["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # Task 7: Resume from checkpoint if provided
    resume_path = cfg["checkpointing"].get("resume_from")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f"[Resumed] Loaded checkpoint from {resume_path} at epoch {ckpt['epoch']}. Continuing from epoch {start_epoch}.")

    total_epochs = max_epochs_override or cfg["training"]["epochs"]
    accum_steps = cfg["training"]["gradient_accumulation_steps"]

    with mlflow.start_run(run_name=run_name):
        # Log flattened config parameters
        mlflow.log_params({
            "seed": cfg["seed"],
            "batch_size": cfg["training"]["batch_size"],
            "effective_batch_size": cfg["training"]["batch_size"] * accum_steps,
            "accum_steps": accum_steps,
            "num_workers": cfg["training"]["num_workers"],
            "optimizer": cfg["training"]["optimizer"],
            "scheduler": cfg["training"]["scheduler"],
            "lr": cfg["training"]["lr"]
        })

        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accum_steps
                loss.backward()
                running_loss += loss.item() * accum_steps

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100.0 * correct / total

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            }, step=epoch)

            print(f"Epoch [{epoch}/{total_epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save Checkpoint
            ckpt_path = os.path.join(cfg["checkpointing"]["save_dir"], f"ckpt_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "history": history
            }, ckpt_path)

        # Log final checkpoint as MLflow artifact
        mlflow.log_artifact(ckpt_path)

    return history

# ==============================================================================
# Task 2: Profile One Epoch with torch.profiler
# ==============================================================================
def profile_one_epoch(cfg):
    print("\n--- Task 2: Profiling One Epoch ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_dataloaders(cfg)
    model = CompactCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    phase_times = {"data_loading": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}

    with profile(activities=activities, record_shapes=True) as prof:
        model.train()
        data_iter = iter(train_loader)
        for _ in range(len(train_loader)):
            t0 = time.perf_counter()
            with record_function("data_loading"):
                inputs, targets = next(data_iter)
                inputs, targets = inputs.to(device), targets.to(device)
            t1 = time.perf_counter()

            with record_function("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            t2 = time.perf_counter()

            with record_function("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
            t3 = time.perf_counter()

            with record_function("optimizer_step"):
                optimizer.step()
            t4 = time.perf_counter()

            phase_times["data_loading"] += (t1 - t0)
            phase_times["forward"] += (t2 - t1)
            phase_times["backward"] += (t3 - t2)
            phase_times["optimizer"] += (t4 - t3)

    total_time = sum(phase_times.values())
    print(f"Total Epoch Time: {total_time:.4f}s")
    for k, v in phase_times.items():
        print(f"  {k:<15}: {v:.4f}s ({(v / total_time) * 100:.2f}%)")
    
    print("\nTop PyTorch Profiler Operators:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=6))

# ==============================================================================
# Task 3: Vary num_workers [0, 1, 2, 4, 8] & Plot Bottleneck
# ==============================================================================
def study_num_workers(cfg):
    print("\n--- Task 3: Worker Count Scaling Study ---")
    worker_counts = [0, 1, 2, 4, 8]
    epoch_times = []

    for nw in worker_counts:
        loader, _ = get_dataloaders(cfg, num_workers=nw)
        start = time.perf_counter()
        for _ in loader:
            pass
        elapsed = time.perf_counter() - start
        epoch_times.append(elapsed)
        print(f"num_workers={nw}: Data iteration time = {elapsed:.4f}s")

    plt.figure(figsize=(7, 4))
    plt.plot(worker_counts, epoch_times, marker='o', linewidth=2, color='#1f77b4')
    plt.title("DataLoader Epoch Iteration Time vs. num_workers")
    plt.xlabel("num_workers")
    plt.ylabel("Epoch Data Loading Time (seconds)")
    plt.xticks(worker_counts)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("num_workers_bottleneck.png", dpi=150)
    print("Saved plot to num_workers_bottleneck.png")

# ==============================================================================
# Task 4: Batch Size Study [32, 64, 128, 256, 512]
# ==============================================================================
def study_batch_sizes(cfg):
    print("\n--- Task 4: Batch Size Study (5 Epochs) ---")
    batch_sizes = [32, 64, 128, 256, 512]
    results = []

    for bs in batch_sizes:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["training"]["batch_size"] = bs
        run_cfg["training"]["epochs"] = 5
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        hist = train_pipeline(run_cfg, max_epochs_override=5, run_name_override=f"bs_{bs}")
        elapsed = time.perf_counter() - t0
        
        num_samples = int(60000 * (1 - cfg["dataset"]["val_split"])) * 5
        throughput = num_samples / elapsed
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        final_acc = hist["val_acc"][-1]
        
        results.append((bs, throughput, peak_mem_mb, final_acc))

    print("\n| Batch Size | Throughput (samples/s) | Peak GPU Memory (MB) | Final Val Acc (5 Epochs) |")
    print("|------------|------------------------|----------------------|--------------------------|")
    for bs, tp, mem, acc in results:
        print(f"| {bs:<10} | {tp:<22.1f} | {mem:<20.2f} | {acc:<23.2f}% |")

# ==============================================================================
# Task 5: Schedulers & Optimizers Comparison (10 Epochs)
# ==============================================================================
def study_schedulers_and_optimizers(cfg):
    print("\n--- Task 5: Optimizers & LR Schedulers Comparison (10 Epochs) ---")
    configs = [
        ("adam", "step"),
        ("adam", "cosine"),
        ("adam", "plateau"),
        ("sgd_momentum", "step"),
        ("sgd_momentum", "cosine"),
        ("sgd_momentum", "plateau"),
    ]
    summary = []
    for opt, sched in configs:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["training"]["optimizer"] = opt
        run_cfg["training"]["scheduler"] = sched
        run_cfg["training"]["epochs"] = 10
        hist = train_pipeline(run_cfg, max_epochs_override=10, run_name_override=f"{opt}_{sched}")
        summary.append((opt, sched, max(hist["val_acc"]), hist["val_loss"][-1]))

    print("\n| Optimizer    | Scheduler | Best Val Acc (%) | Final Val Loss |")
    print("|--------------|-----------|------------------|----------------|")
    for opt, sched, acc, loss in summary:
        print(f"| {opt:<12} | {sched:<9} | {acc:<16.2f} | {loss:<14.4f} |")

# ==============================================================================
# Task 6: Gradient Accumulation Verification (BS 256 vs BS 64 x 4)
# ==============================================================================
def verify_gradient_accumulation(cfg):
    print("\n--- Task 6: Gradient Accumulation Verification ---")
    cfg_256 = copy.deepcopy(cfg)
    cfg_256["training"]["batch_size"] = 256
    cfg_256["training"]["gradient_accumulation_steps"] = 1
    hist_256 = train_pipeline(cfg_256, max_epochs_override=5, run_name_override="bs256_native")

    cfg_accum = copy.deepcopy(cfg)
    cfg_accum["training"]["batch_size"] = 64
    cfg_accum["training"]["gradient_accumulation_steps"] = 4
    hist_accum = train_pipeline(cfg_accum, max_epochs_override=5, run_name_override="bs64_accum4")

    print("\nEpoch-by-Epoch Validation Accuracy Comparison:")
    for ep in range(5):
        print(f"Epoch {ep+1}: Native BS=256 Acc = {hist_256['val_acc'][ep]:.2f}% | Accumulated BS=64x4 Acc = {hist_accum['val_acc'][ep]:.2f}%")

# ==============================================================================
# Task 7: Interrupt at Epoch 3 & Resume
# ==============================================================================
def test_interrupt_and_resume(cfg):
    print("\n--- Task 7: Interrupt at Epoch 3 & Resume ---")
    # Phase 1: Train up to epoch 3
    cfg_part1 = copy.deepcopy(cfg)
    cfg_part1["checkpointing"]["resume_from"] = None
    hist_part1 = train_pipeline(cfg_part1, max_epochs_override=3, run_name_override="interrupt_part1")

    # Phase 2: Resume from epoch 3 checkpoint to epoch 6
    cfg_part2 = copy.deepcopy(cfg)
    cfg_part2["checkpointing"]["resume_from"] = os.path.join(cfg["checkpointing"]["save_dir"], "ckpt_epoch_3.pt")
    hist_full = train_pipeline(cfg_part2, max_epochs_override=6, run_name_override="resume_part2")

    print("\nContinuous Validation Loss Curve Across Interruption (Epochs 1 -> 6):")
    for ep, loss in enumerate(hist_full["val_loss"], 1):
        marker = " <-- Resumed after this epoch" if ep == 3 else ""
        print(f"  Epoch {ep}: Val Loss = {loss:.4f}, Val Acc = {hist_full['val_acc'][ep-1]:.2f}%{marker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 6 training experiments.")
    parser.add_argument(
        "--task",
        choices=("profile", "workers", "batch-sizes", "schedulers", "accumulation", "resume", "all"),
        default="profile",
        help="Experiment to run (default: profile). Use 'all' for the complete suite.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "config.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    resolve_config_paths(config, os.path.dirname(config_path))

    if args.task in ("profile", "all"):
        profile_one_epoch(config)
    if args.task in ("workers", "all"):
        study_num_workers(config)
    if args.task in ("batch-sizes", "all"):
        study_batch_sizes(config)
    if args.task in ("schedulers", "all"):
        study_schedulers_and_optimizers(config)
    if args.task in ("accumulation", "all"):
        verify_gradient_accumulation(config)
    if args.task in ("resume", "all"):
        test_interrupt_and_resume(config)