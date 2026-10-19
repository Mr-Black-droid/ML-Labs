import sys
from importlib.metadata import version
import pathlib

# Ensure results directory exists
results_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

version_file = results_dir / "versions.txt"
packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchinfo": "torchinfo",
    "thop": "thop",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "mlflow": "mlflow",
    "memory-profiler": "memory-profiler",
    "psutil": "psutil",
    "codecarbon": "codecarbon",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pytest": "pytest",
    "httpx": "httpx",
    "locust": "locust",
    "requests": "requests",
    "pyarrow": "pyarrow",
    "joblib": "joblib",
    "tqdm": "tqdm",
}

with version_file.open("w") as file:
    file.write(f"Python: {sys.version}\n")
    for package_name, distribution_name in packages.items():
        file.write(f"{package_name}: {version(distribution_name)}\n")

print(f"Versions successfully saved to {version_file}")