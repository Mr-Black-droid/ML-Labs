import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import CompactStudentCNN
from .utils import evaluate_accuracy, measure_cpu_latency_ms, get_disk_size_mb, count_parameters


def train_with_distillation(student: nn.Module, teacher: nn.Module, train_loader, device,
                            epochs: int = 8, lr: float = 1e-3, T: float = 4.0, alpha: float = 0.5):
    """
    Loss = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))
    """
    student.to(device).train()
    teacher.to(device).eval()

    optimizer = optim.Adam(student.parameters(), lr=lr)
    ce_criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)

            hard_loss = ce_criterion(student_logits, y)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean"
            )
            loss = alpha * hard_loss + (1.0 - alpha) * (T ** 2) * soft_loss

            loss.backward()
            optimizer.step()


def execute_distillation_experiments(teacher_model, train_loader, test_loader, device):
    epochs = 8

    # 1. Plain Student (without teacher)
    plain_student = CompactStudentCNN().to(device)
    opt = optim.Adam(plain_student.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    plain_student.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(plain_student(x), y)
            loss.backward()
            opt.step()

    plain_acc = evaluate_accuracy(plain_student, test_loader, device)
    plain_size = get_disk_size_mb(plain_student, "plain_student.pt")
    plain_lat = measure_cpu_latency_ms(copy.deepcopy(plain_student))

    # 2. Distilled Student (with teacher, T=4, alpha=0.5)
    distilled_student = CompactStudentCNN().to(device)
    train_with_distillation(distilled_student, teacher_model, train_loader, device, epochs=epochs, T=4.0, alpha=0.5)

    distill_acc = evaluate_accuracy(distilled_student, test_loader, device)
    distill_size = get_disk_size_mb(distilled_student, "distilled_student.pt")
    distill_lat = measure_cpu_latency_ms(copy.deepcopy(distilled_student))

    params = count_parameters(plain_student)
    print(f"\n[Task 4] Student Comparison ({params:,} parameters):")
    print(f"  Plain Student     -> Acc: {plain_acc:.2f}% | Size: {plain_size:.3f} MB | Latency: {plain_lat:.3f} ms")
    print(f"  Distilled Student -> Acc: {distill_acc:.2f}% | Size: {distill_size:.3f} MB | Latency: {distill_lat:.3f} ms")

    return {
        "student_params": params,
        "plain_acc": plain_acc,
        "plain_size": plain_size,
        "plain_lat": plain_lat,
        "distill_acc": distill_acc,
        "distill_size": distill_size,
        "distill_lat": distill_lat,
    }