import copy
import torch

if __package__:
    from .src.models import OversizedTeacherCNN
    from .src.utils import (
        set_seed,
        get_fashion_mnist_loaders,
        train_standard,
        evaluate_accuracy,
        count_parameters,
        get_disk_size_mb,
        measure_cpu_latency_ms,
    )
    from .src.pruning import execute_pruning_experiments
    from .src.distillation import execute_distillation_experiments
else:
    from src.models import OversizedTeacherCNN
    from src.utils import (
        set_seed,
        get_fashion_mnist_loaders,
        train_standard,
        evaluate_accuracy,
        count_parameters,
        get_disk_size_mb,
        measure_cpu_latency_ms,
    )
    from src.pruning import execute_pruning_experiments
    from src.distillation import execute_distillation_experiments


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)

    # Task 1: Oversized Baseline (~1M params)
    print("=== Task 1: Training Oversized Baseline CNN ===")
    baseline = OversizedTeacherCNN().to(device)
    train_standard(baseline, train_loader, device, epochs=8, lr=1e-3)

    base_params = count_parameters(baseline)
    base_acc = evaluate_accuracy(baseline, test_loader, device)
    base_size = get_disk_size_mb(baseline, "baseline.pt")
    base_lat = measure_cpu_latency_ms(copy.deepcopy(baseline))

    print(f"Baseline -> Acc: {base_acc:.2f}% | Params: {base_params:,} | Size: {base_size:.3f} MB | CPU Latency: {base_lat:.3f} ms\n")

    # Tasks 2 & 3: Magnitude & Structured Pruning + Latency/Storage Analysis
    print("=== Tasks 2 & 3: Pruning & Latency Analysis ===")
    execute_pruning_experiments(baseline, train_loader, test_loader, device)

    # Task 4: Knowledge Distillation
    print("\n=== Task 4: Knowledge Distillation ===")
    execute_distillation_experiments(baseline, train_loader, test_loader, device)


if __name__ == "__main__":
    main()