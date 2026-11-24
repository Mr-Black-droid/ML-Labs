# Lab 01 Report

## 1. Goal
To train baseline linear (Logistic Regression) and ensemble (Random Forest) models on the Breast Cancer Wisconsin dataset, measure their system costs (training time, latency, model size, and peak memory), and evaluate their deployment suitability against Cloud, Edge, Mobile, and TinyML hardware budgets.

## 2. Method
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) loaded via scikit-learn (`load_breast_cancer`).
- **Data Split:** 70/30 stratified split with `random_state=42`.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Profiling:** Measured median training time across 5 runs, single-sample inference latency over 100 iterations, serialized model size via `joblib.dump`, and RSS peak memory via `psutil`.

## 3. Results

The benchmarks were run with Python 3.10.11 and the package versions recorded
by `src/print_versions.py` in `results/versions.txt`.

| Model | Accuracy | Median training time (s) | Median inference latency (ms) | Model size (KB) | Peak RSS memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9415 (94.15%) | 0.6499 | 0.2560 | 1.03 | 135.39 |
| Random Forest | 0.9357 (93.57%) | 0.5677 | 13.6849 | 284.07 | 136.25 |

The complete generated data, including model sizes in bytes, is available in
`results/baseline_accuracy.csv`.

The run used these key dependency versions:

- `numpy` 1.26.4, `pandas` 2.2.2, `scikit-learn` 1.5.0, and `scipy` 1.13.1
- `joblib` 1.4.2 and `psutil` 5.9.8
- `torch` 2.3.1 and `torchvision` 0.18.1

## 4. Conclusions
1. Logistic Regression produces the much smaller artifact and has substantially lower inference latency, although the measured Random Forest training time is slightly lower in this run.
2. Logistic Regression achieved 94.15% accuracy and Random Forest achieved 93.57%; therefore, neither model exceeded 95% accuracy in this run.
3. The measured process RSS was approximately 135–136 MB for both models, so neither fits within a strict TinyML runtime-memory budget of less than 256 KB. The Random Forest artifact is also much larger at 284.07 KB.

## 5. Deployment Suitability
Based on the measured artifact sizes and inference latencies, both models satisfy the stated broad **Cloud, Edge, and Mobile** thresholds of less than 10 MB and 20 ms, respectively. Logistic Regression is the better choice for resource-constrained deployment because its artifact is only 1.03 KB and its median latency is 0.2560 ms. The measured process memory footprint should still be considered when targeting a specific device.