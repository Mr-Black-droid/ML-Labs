import time
import os
import pathlib
import psutil
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure results directory exists next to the lab directory, regardless of cwd.
results_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
X, y = load_breast_cancer(return_X_y=True)

# 2. Split 70/30 stratified with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
process = psutil.Process(os.getpid())

for name, model in models.items():
    # 3. Training time: median of 5 runs after 1 warm-up
    model.fit(X_train, y_train)  # Warm-up run
    
    train_times = []
    for _ in range(5):
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        train_times.append(end_time - start_time)
    
    median_train_time = np.median(train_times)
    
    # Peak memory tracking via psutil
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # Single-sample inference latency: repeat 100 times, report median
    single_sample = X_test[:1]
    latencies = []
    for _ in range(100):
        inf_start = time.perf_counter()
        _ = model.predict(single_sample)
        inf_end = time.perf_counter()
        latencies.append((inf_end - inf_start) * 1000)  # Convert to milliseconds
    median_latency = np.median(latencies)
    
    # Save model artifacts
    model_path = results_dir / f"{name.lower()}_model.joblib"
    joblib.dump(model, model_path)
    
    # Save primary model to results/model.joblib as required
    if name == "RandomForest":
        joblib.dump(model, results_dir / "model.joblib")

    model_size_bytes = os.path.getsize(model_path)
    model_size_kb = model_size_bytes / 1024
    
    # Test accuracy to 4 decimals
    preds = model.predict(X_test)
    acc = round(accuracy_score(y_test, preds), 4)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Training Time Median (s)": round(median_train_time, 4),
        "Inference Latency Median (ms)": round(median_latency, 4),
        "Model Size (Bytes)": model_size_bytes,
        "Model Size (KB)": round(model_size_kb, 2),
        "Peak Memory (MB)": round(peak_memory_mb, 2)
    })

# Save baseline accuracies to CSV
df_results = pd.DataFrame(results)
df_results.to_csv(results_dir / "baseline_accuracy.csv", index=False)
print("Training and benchmarking complete.")
print(df_results)