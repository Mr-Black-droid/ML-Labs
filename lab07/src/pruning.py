import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from .utils import evaluate_accuracy, measure_cpu_latency_ms, get_disk_size_mb, count_parameters


def _get_prunable_layers(model: nn.Module):
    return [(m, "weight") for _, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def run_unstructured_pruning(base_model, sparsity: float, mode: str, train_loader, test_loader, device):
    """
    Masks smallest weights by absolute value (globally or per-layer) and fine-tunes 3 epochs with mask fixed.
    """
    model = copy.deepcopy(base_model).to(device)
    layers = _get_prunable_layers(model)

    if sparsity > 0.0:
        if mode == "global":
            prune.global_unstructured(layers, pruning_method=prune.L1Unstructured, amount=sparsity)
        elif mode == "layer":
            for mod, name in layers:
                prune.l1_unstructured(mod, name=name, amount=sparsity)
        else:
            raise ValueError("mode must be 'global' or 'layer'")

        # Fine-tune 3 epochs with pruning mask fixed (PyTorch forward pre-hook enforces mask during training)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(3):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # Make zeros permanent in weight tensors
        for mod, name in layers:
            prune.remove(mod, name)

    acc = evaluate_accuracy(model, test_loader, device)
    return model, acc


def physically_prune_filters_l2(base_model, prune_ratio: float, train_loader, test_loader, device):
    """
    Removes whole Conv2d filters ranked by L2 norm at prune_ratio in [0.10, 0.25, 0.50],
    rebuilding conv1, bn1, conv2, bn2, and fc1 input features so tensor shapes actually shrink.
    """
    model = copy.deepcopy(base_model).to("cpu")

    # 1. Rank conv1 filters by L2 norm across (in_channels, kH, kW)
    c1_norms = torch.norm(model.conv1.weight.data.view(model.conv1.out_channels, -1), p=2, dim=1)
    k1 = int(round(model.conv1.out_channels * (1.0 - prune_ratio)))
    keep1 = torch.topk(c1_norms, k1, largest=True).indices.sort().values

    # 2. Rank conv2 filters by L2 norm
    c2_norms = torch.norm(model.conv2.weight.data.view(model.conv2.out_channels, -1), p=2, dim=1)
    k2 = int(round(model.conv2.out_channels * (1.0 - prune_ratio)))
    keep2 = torch.topk(c2_norms, k2, largest=True).indices.sort().values

    # Rebuild conv1 & bn1
    new_conv1 = nn.Conv2d(1, k1, kernel_size=3, padding=1)
    new_conv1.weight.data = model.conv1.weight.data[keep1].clone()
    new_conv1.bias.data = model.conv1.bias.data[keep1].clone()

    new_bn1 = nn.BatchNorm2d(k1)
    new_bn1.weight.data = model.bn1.weight.data[keep1].clone()
    new_bn1.bias.data = model.bn1.bias.data[keep1].clone()
    new_bn1.running_mean = model.bn1.running_mean[keep1].clone()
    new_bn1.running_var = model.bn1.running_var[keep1].clone()

    # Rebuild conv2 & bn2 (slicing both output filters keep2 and input channels keep1)
    new_conv2 = nn.Conv2d(k1, k2, kernel_size=3, padding=1)
    new_conv2.weight.data = model.conv2.weight.data[keep2][:, keep1].clone()
    new_conv2.bias.data = model.conv2.bias.data[keep2].clone()

    new_bn2 = nn.BatchNorm2d(k2)
    new_bn2.weight.data = model.bn2.weight.data[keep2].clone()
    new_bn2.bias.data = model.bn2.bias.data[keep2].clone()
    new_bn2.running_mean = model.bn2.running_mean[keep2].clone()
    new_bn2.running_var = model.bn2.running_var[keep2].clone()

    # Rebuild fc1 input columns matching kept conv2 channels * (7 * 7) spatial grid
    fc1_in_idx = torch.cat([torch.arange(c * 49, (c + 1) * 49) for c in keep2])
    new_fc1 = nn.Linear(k2 * 49, model.fc1.out_features)
    new_fc1.weight.data = model.fc1.weight.data[:, fc1_in_idx].clone()
    new_fc1.bias.data = model.fc1.bias.data.clone()

    model.conv1, model.bn1 = new_conv1, new_bn1
    model.conv2, model.bn2 = new_conv2, new_bn2
    model.fc1 = new_fc1
    model.to(device)

    # Fine-tune 3 epochs after filter removal
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(3):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    acc = evaluate_accuracy(model, test_loader, device)
    return model, acc


def execute_pruning_experiments(baseline_model, train_loader, test_loader, device):
    sparsities = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
    global_accs, layer_accs = [], []
    models_global = {}

    for s in sparsities:
        m_g, acc_g = run_unstructured_pruning(baseline_model, s, "global", train_loader, test_loader, device)
        _, acc_l = run_unstructured_pruning(baseline_model, s, "layer", train_loader, test_loader, device)
        global_accs.append(acc_g)
        layer_accs.append(acc_l)
        models_global[s] = m_g
        print(f"Unstructured Sparsity {s:>4.2f} -> Global Acc: {acc_g:.2f}% | Per-Layer Acc: {acc_l:.2f}%")

    struct_ratios = [0.10, 0.25, 0.50]
    struct_accs, struct_latencies = [], []
    for r in struct_ratios:
        m_s, acc_s = physically_prune_filters_l2(baseline_model, r, train_loader, test_loader, device)
        lat_s = measure_cpu_latency_ms(copy.deepcopy(m_s))
        struct_accs.append(acc_s)
        struct_latencies.append(lat_s)
        print(
            f"Structured L2 Filter Pruning {int(r*100):>2d}% -> "
            f"Acc: {acc_s:.2f}% | Params: {count_parameters(m_s):,} | CPU Latency: {lat_s:.3f} ms"
        )

    # Plot Accuracy vs Sparsity for all three curves
    plt.figure(figsize=(8.5, 5))
    plt.plot(sparsities, global_accs, marker="o", linewidth=2, label="Unstructured Global (L1)")
    plt.plot(sparsities, layer_accs, marker="s", linewidth=2, label="Unstructured Per-Layer (L1)")
    plt.plot(struct_ratios, struct_accs, marker="^", linewidth=2, linestyle="--", label="Structured Filter Pruning (L2)")
    plt.title("Fashion-MNIST Accuracy vs. Sparsity (Fine-Tuned 3 Epochs)")
    plt.xlabel("Sparsity / Filter Removal Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sparsity_curves.png", dpi=150)

    # Dense vs Sparse Size & Latency Analysis on 90% pruned model
    m90 = models_global[0.90]
    dense_mb = get_disk_size_mb(m90, "dense_90.pt", sparse_format=False)
    sparse_mb = get_disk_size_mb(m90, "sparse_90.pt", sparse_format=True)
    dense_lat = measure_cpu_latency_ms(copy.deepcopy(m90))

    print(f"\n[Task 3] 90% Unstructured Sparsity Storage & Latency:")
    print(f"  Dense Disk Size : {dense_mb:.3f} MB | Sparse Disk Size: {sparse_mb:.3f} MB")
    print(f"  Dense CPU Latency: {dense_lat:.3f} ms (unchanged vs unpruned dense baseline)")

    return {
        "sparsities": sparsities,
        "global_accs": global_accs,
        "layer_accs": layer_accs,
        "struct_ratios": struct_ratios,
        "struct_accs": struct_accs,
        "struct_latencies": struct_latencies,
        "dense_mb_90": dense_mb,
        "sparse_mb_90": sparse_mb,
        "dense_lat_90": dense_lat,
    }