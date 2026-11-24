# Lab 2: A Neural Network from Scratch - Report

## 1. Goal
To design and implement a two-hidden-layer neural network completely from scratch using NumPy for the Optical Recognition of Handwritten Digits (`load_digits`) dataset, validate manual backpropagation using numerical gradient checks, evaluate performance across various hyperparameter settings, and analyze computational timing characteristics.

## 2. Method
- **Dataset Preparation**: Loaded the digits dataset, scaled feature inputs by dividing by 16, one-hot encoded the 10-class targets, and executed a 70/30 stratified split with `random_state=42`.
- **Architecture**: Constructed a multi-layer perceptron comprising an input dimension of 64, two hidden layers with widths $W \in \{16, 64\}$ activated by ReLU functions, and an output layer with 10 units activated via Softmax. Weights were initialized using $\mathcal{N}(0, 0.01)$ and biases set to zero.
- **Optimization & Loss**: Implemented categorical cross-entropy loss along with manual chain-rule gradient derivations for all layers. Verified correctness using finite-difference numerical gradient checking ($\epsilon = 10^{-5}$).
- **Evaluation**: Trained models across learning rates $\{0.001, 0.01, 0.1\}$ and widths $\{16, 64\}$ using mini-batch gradient descent (batch size 32) over 50 epochs. Benchmarked forward and backward pass speeds using median execution times over 100 runs following a warm-up phase.

## 3. Results Table or Plot

| Hidden Width ($W$) | Learning Rate ($\alpha$) | Final Test Accuracy (%) |
| :--- | :--- | :--- |
| 16 | 0.001 | 85.19% |
| 16 | 0.01  | 92.59% |
| 16 | 0.1   | 94.07% |
| 64 | 0.001 | 89.81% |
| 64 | 0.01  | 95.37% |
| 64 | 0.1   | 96.30% |

![Training Loss Curves](../results/loss_curves.png)

## 4. Three Conclusions in Your Own Words
1. **Hyperparameter Impact**: Increasing both hidden layer width and learning rate (up to 0.1) substantially accelerated convergence speed and improved final test accuracy, with the wider network ($W=64$) capturing richer representations from the digit feature space.
2. **Gradient Verification**: The manual chain-rule derivations successfully achieved alignment with numerical approximations (passing gradient checks with relative errors well below $10^{-6}$), confirming the mathematical soundness of the backpropagation logic.
3. **Execution Bottlenecks**: Timing benchmarks reveal that the backward pass requires roughly twice the computational duration of the forward pass, directly reflecting the increased matrix multiplication workload necessary for both error propagation and weight gradient accumulation.