import os
import random
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Python Version: {sys.version}")
print(f"NumPy Version: {np.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"Matplotlib Version: {matplotlib.__version__}")
print(f"PyTorch Version: {torch.__version__}")



def prepare_data():
    digits = load_digits()
    X, y = digits.data, digits.target

    X = X / 16.0

    num_classes = 10
    y_onehot = np.eye(num_classes)[y]

    # 70/30 stratified split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


class NeuralNetwork:

    def __init__(self, input_dim=64, hidden_dim=16, output_dim=10):
        # Initialize weights from N(0, 0.01) and biases to zeros
        self.W1 = np.random.normal(0, 0.01, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.normal(0, 0.01, size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros((1, hidden_dim))

        self.W3 = np.random.normal(0, 0.01, size=(hidden_dim, output_dim))
        self.b3 = np.zeros((1, output_dim))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def compute_loss(self, Y_hat, Y):
        eps = 1e-15
        Y_hat_clipped = np.clip(Y_hat, eps, 1 - eps)
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_hat_clipped)) / m

    def backward(self, X, Y):
        m = X.shape[0]

        dZ3 = (self.A3 - Y) / m
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def update_params(self, grads, lr):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]
        self.W3 -= lr * grads["dW3"]
        self.b3 -= lr * grads["db3"]



def run_validations(X_train, y_train):
    nn = NeuralNetwork(hidden_dim=16)
    sample_X = X_train[:100]
    sample_y = y_train[:100]

    # Untrained loss check
    preds = nn.forward(sample_X)
    loss = nn.compute_loss(preds, sample_y)
    print(f"Sanity Check - Untrained Loss: {loss:.4f} (Expected ≈ 2.3026)")

    assert preds.shape == (
        100,
        10,
    ), f"Expected shape (100, 10), got {preds.shape}"
    row_sums = np.sum(preds, axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-6
    ), "Softwmax rows do not sum to 1 within 1e-6."

    X_small = X_train[:32]
    Y_small = y_train[:32]
    nn.forward(X_small)
    grads = nn.backward(X_small, Y_small)

    eps = 1e-5
    passed_checks = 0
    param_targets = [("W1", nn.W1, grads["dW1"]), ("W3", nn.W3, grads["dW3"])]

    for i in range(10):
        _, W, dW = param_targets[i % len(param_targets)]
        idx = (np.random.randint(0, W.shape[0]), np.random.randint(0, W.shape[1]))

        W[idx] += eps
        loss_plus = nn.compute_loss(nn.forward(X_small), Y_small)
        W[idx] -= 2 * eps
        loss_minus = nn.compute_loss(nn.forward(X_small), Y_small)
        W[idx] += eps

        num_grad = (loss_plus - loss_minus) / (2 * eps)
        ana_grad = dW[idx]
        rel_error = np.abs(ana_grad - num_grad) / (
            np.abs(ana_grad) + np.abs(num_grad) + 1e-8
        )

        if rel_error < 1e-6:
            passed_checks += 1

    print(
        f"Gradient Check: {passed_checks}/10 weights passed with relative error < 1e-6."
    )



def run_training_loop(X_train, X_test, y_train, y_test):
    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    epochs = 50
    batch_size = 32

    os.makedirs("lab02/results", exist_ok=True)
    plt.figure(figsize=(10, 6))

    for W in widths:
        for lr in learning_rates:
            nn = NeuralNetwork(hidden_dim=W)
            train_losses = []

            for _ in range(epochs):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                epoch_loss = 0.0
                num_batches = int(np.ceil(X_train.shape[0] / batch_size))

                for b in range(num_batches):
                    X_b = X_shuffled[b * batch_size : (b + 1) * batch_size]
                    y_b = y_shuffled[b * batch_size : (b + 1) * batch_size]

                    y_hat = nn.forward(X_b)
                    loss = nn.compute_loss(y_hat, y_b)
                    grads = nn.backward(X_b, y_b)
                    nn.update_params(grads, lr)

                    epoch_loss += loss * len(X_b)

                epoch_loss /= X_train.shape[0]
                train_losses.append(epoch_loss)

            test_preds = nn.forward(X_test)
            test_acc = np.mean(
                np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1)
            )
            print(
                f"Width: {W:2d} | LR: {lr:5f} | Final Test Accuracy: {test_acc*100:.2f}%"
            )

            plt.plot(
                range(1, epochs + 1),
                train_losses,
                label=f"W={W}, lr={lr} (Acc:{test_acc:.2f})",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Lab 2 Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("lab02/results/loss_curves.png")
    plt.close()
    print("Saved training loss plot to lab02/results/loss_curves.png")



def run_timing_analysis(X_train):
    nn = NeuralNetwork(hidden_dim=64)
    sample_batch = X_train[:32]
    y_dummy = np.zeros((32, 10))

    _ = nn.forward(sample_batch)
    _ = nn.backward(sample_batch, y_dummy)

    forward_times = []
    backward_times = []

    for _ in range(100):
        t0 = time.perf_counter()
        _ = nn.forward(sample_batch)
        t1 = time.perf_counter()
        _ = nn.backward(sample_batch, y_dummy)
        t2 = time.perf_counter()

        forward_times.append((t1 - t0) * 1000)
        backward_times.append((t2 - t1) * 1000)

    med_fwd = np.median(forward_times)
    med_bwd = np.median(backward_times)

    print(f"Median Forward Pass Time: {med_fwd:.4f} ms")
    print(f"Median Backward Pass Time: {med_bwd:.4f} ms")
    print(
        f"Dominant Pass: {'Backward' if med_bwd > med_fwd else 'Forward'} ({med_bwd/med_fwd:.2f}x ratio)"
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    run_validations(X_train, y_train)
    run_training_loop(X_train, X_test, y_train, y_test)
    run_timing_analysis(X_train)