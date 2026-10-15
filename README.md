# 🔢 Handwritten Digits Recognition

A comprehensive comparative study and practical implementation of handwritten digit classification using classic Machine Learning algorithms and Deep Learning architectures. This repository explores implementations ranging from **first-principles mathematical modeling from scratch in NumPy** to **production-grade pipelines in Scikit-Learn and PyTorch**.

---

## 📌 Project Overview

This repository benchmarks and contrasts multiple approaches for recognizing handwritten digits:
1. **Classical Machine Learning**: Support Vector Machines (SVM) and Multiclass Logistic Regression using [Scikit-Learn](https://scikit-learn.org/).
2. **Deep Learning from Scratch**: A 2-layer Neural Network implemented purely using [NumPy](https://numpy.org/)—including manual forward propagation, softmax cross-entropy loss computation, analytical backpropagation, and mini-batch gradient descent.
3. **Modern Deep Learning**: A Multi-Layer Perceptron (MLP) built and optimized with [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/).
4. **PyTorch Foundations**: Exploration of tensor computations, automatic differentiation (`autograd`), and GPU/CUDA acceleration.

---

## 📊 Performance & Comparison

| Approach / Model | Framework | Dataset | Architecture / Configuration | Test Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| **Support Vector Classifier (SVC)** | Scikit-Learn | Digits (8×8) | RBF Kernel (`C=1.0`, `gamma='scale'`) | **99.17%** |
| **PyTorch Multi-Layer Perceptron** | PyTorch | MNIST (28×28) | `784` $\rightarrow$ `128` (ReLU) $\rightarrow$ `10` (Adam, 10 Epochs) | **97.62%** |
| **Multiclass Logistic Regression** | Scikit-Learn | Digits (8×8) | L-BFGS solver (`max_iter=1000`) | **95.56%** |
| **Neural Network from Scratch** | NumPy | MNIST (28×28) | `784` $\rightarrow$ `128` (ReLU) $\rightarrow$ `10`, Mini-batch SGD (5 Epochs) | **92.02%** |

---

## 📁 Repository Structure

```text
digits_recognition/
├── data/
│   └── MNIST/
│       └── raw/                     # Raw MNIST binary dataset files (.ubyte / .gz)
├── models/                          # Directory for exported model checkpoints & weights
│   └── .gitkeep
├── numpy_nn.ipynb                   # Neural Network implemented from scratch using NumPy
├── pytorch_basics.ipynb             # PyTorch fundamentals: Tensors, autograd, CUDA setup
├── pytorch_nn.ipynb                 # PyTorch MLP trained on MNIST using DataLoader & Adam
├── scikit_learn_logistic_reg.ipynb  # Logistic Regression classification on 8x8 Digits
├── scikit_learn_svm.ipynb           # SVM (RBF kernel) classifier with Confusion Matrix
├── main.ipynb                       # Playground / Scratch execution notebook
├── .gitignore                       # Ignored files (virtualenvs, checkpoints, binaries)
└── README.md                        # Documentation and project guide
```

---

## 🔬 Notebook Highlights

### 1. [`numpy_nn.ipynb`](file:///home/shadow/Desktop/digits_recognition/numpy_nn.ipynb) — Neural Network from Scratch
- **No high-level autograd or DL framework**: Implemented purely with NumPy arrays and vector math.
- **Forward Pass**:
  $$Z^{[1]} = X W^{[1]} + b^{[1]}$$
  $$A^{[1]} = \text{ReLU}(Z^{[1]})$$
  $$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}$$
  $$\hat{Y} = \text{Softmax}(Z^{[2]})$$
- **Loss Computation**: Numerically stable Softmax Cross-Entropy loss.
- **Backward Pass**: Full manual chain rule derivation for $\frac{\partial \mathcal{L}}{\partial W^{[2]}}$, $\frac{\partial \mathcal{L}}{\partial b^{[2]}}$, $\frac{\partial \mathcal{L}}{\partial W^{[1]}}$, and $\frac{\partial \mathcal{L}}{\partial b^{[1]}}$.
- **Training**: Mini-batch Gradient Descent with batch size 64 over 5 epochs reaching **92.02%** test accuracy.

### 2. [`pytorch_nn.ipynb`](file:///home/shadow/Desktop/digits_recognition/pytorch_nn.ipynb) — PyTorch Neural Network
- Uses `torch.nn.Sequential` for clean, modular layer definitions:
  - `nn.Flatten()` (flattens 28×28 image tensors to 784-dimensional vectors)
  - `nn.Linear(784, 128)` followed by `nn.ReLU()`
  - `nn.Linear(128, 10)` output logits
- PyTorch `DataLoader` with batch size 64 and shuffling.
- Optimized with `torch.optim.Adam(lr=0.001)` and `nn.CrossEntropyLoss()`.
- Fast convergence achieving **97.62%** test accuracy within 10 epochs.

### 3. [`scikit_learn_svm.ipynb`](file:///home/shadow/Desktop/digits_recognition/scikit_learn_svm.ipynb) — Support Vector Machine
- Evaluates Support Vector Classifier (`SVC`) with an Radial Basis Function (`rbf`) kernel on the Scikit-Learn 8×8 handwritten digits dataset.
- High discriminative boundary separation yielding **99.17%** test accuracy.
- Features classification reports (precision, recall, f1-score) and confusion matrix visualization.

### 4. [`scikit_learn_logistic_reg.ipynb`](file:///home/shadow/Desktop/digits_recognition/scikit_learn_logistic_reg.ipynb) — Logistic Regression
- Softmax / Multinomial Logistic Regression baseline using `sklearn.linear_model.LogisticRegression`.
- Handles multi-class classification directly, converging to **95.56%** accuracy.
- Visual inspection of correctly classified vs. misclassified digits.

### 5. [`pytorch_basics.ipynb`](file:///home/shadow/Desktop/digits_recognition/pytorch_basics.ipynb) — PyTorch Core Concepts
- Hands-on guide to PyTorch tensors, indexing, slicing, broadcasting, and GPU device management (`torch.cuda.is_available()`).

---

## 📦 Datasets

1. **MNIST Database**:
   - 70,000 grayscale handwritten digit images (60,000 training, 10,000 testing).
   - Image resolution: $28 \times 28$ pixels.
   - Classes: 10 digits (`0` through `9`).
   - Automatically loaded and parsed into `./data/MNIST`.

2. **Scikit-Learn Digits**:
   - 1,797 normalized $8 \times 8$ pixel grayscale images.
   - Lightweight dataset ideal for quick prototyping and algorithmic benchmarking.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.11)
- `pip` or standard virtual environment manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/digits_recognition.git
   cd digits_recognition
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision scikit-learn numpy matplotlib jupyterlab
   ```

4. **Launch Jupyter Lab or Notebook**:
   ```bash
   jupyter lab
   ```

---

## 🛠️ Future Improvements

- [ ] Implement a Convolutional Neural Network (CNN / LeNet-5 architecture) in PyTorch to surpass 99.2% on MNIST.
- [ ] Add Data Augmentation (random rotation, affine transformation, slight shear) to improve model generalization.
- [ ] Export trained PyTorch weights (`.pth` / ONNX) into the `models/` directory.
- [ ] Build an interactive drawing canvas using Streamlit or Gradio to test real-time mouse/touch handwritten digit input.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
