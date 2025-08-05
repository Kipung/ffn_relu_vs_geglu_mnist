# 🧠 GeGLU vs ReLU on MNIST (with PyTorch Lightning)

This experiment compares **Feedforward Neural Networks (FFNs)** with different activation types — **ReLU** and **GeGLU** — using the MNIST digit classification dataset. It explores their performance across different hidden dimensions, random hyperparameter trials, and includes statistical bootstrapping to compute confidence intervals.

---

## 🧪 Claim

> **"GeGLU performs better than ReLU."**

This project evaluates that claim on the MNIST dataset using one-epoch training, and varying hidden dimensions and learning configs.

---

## ⚙️ Setup

- **Dataset**: MNIST
- **Models**:
  - `FFN_ReLU`: Uses `ReLU(Linear(x)) → Linear → output`
  - `FFN_GeGLU`: Uses `x_proj * GELU(x_gate)` via `einsum`, modeled after LLaMA's implementation
- **Optimizer**: Adam
- **Batch Sizes**: 8 or 64
- **Learning Rates**: [1e-1, 1e-2, 1e-3, 1e-4]
- **Epochs**: 1 (as per "One Epoch is All You Need")
- **Hidden Dimensions**: [2, 4, 8, 16]
- **Random Trials per k**: `k ∈ {2, 4, 8}`
- **Bootstrap**: 10,000 samples for CI

---

## 🔢 Experiments

### 1. Accuracy vs Hidden Dimension

Compares model performance with increasing hidden dimension.

![Hidden Dim Sweep](./lightning_hidden_dim_sweep.png)

---

### 2. Accuracy Across k Trials

For each `k ∈ {2, 4, 8}`, run `k` random hyperparameter trials and plot best validation accuracy.

- Learning Rate: Random from `[1e-1, 1e-2, 1e-3, 1e-4]`
- Batch Size: Random from `[8, 64]`
- Fixed Hidden Dim: 8

**k = 2**

![Accuracy vs k=2](./lightning_accuracy_vs_k_2.png)

**k = 4**

![Accuracy vs k=4](./lightning_accuracy_vs_k_4.png)

**k = 8**

![Accuracy vs k=8](./lightning_accuracy_vs_k_8.png)

---

### 3. Bootstrap Confidence Intervals (CI)

Bootstrapped 95% confidence intervals for best validation accuracies (per model per k).

**k = 2**

![Bootstrap CI k=2](./lightning_bootstrap_ci_k2.png)

**k = 4**

![Bootstrap CI k=4](./lightning_bootstrap_ci_k4.png)

**k = 8**

![Bootstrap CI k=8](./lightning_bootstrap_ci_k8.png)

---

## 🧐 Conclusion

- **GeGLU consistently outperforms ReLU** across all tested configurations — including varying hidden dimensions and random trial settings (k = 2, 4, 8).
- This **supports the original claim** that GeGLU activations lead to better performance, even under constrained training setups.
- The difference is visible in both:
  - Accuracy vs. Hidden Dimension
  - Bootstrapped 95% Confidence Intervals
- The GELU-based gating in GeGLU appears to help stabilize and enhance performance even with small hidden dimensions and 1-epoch training.

---

## ⚠️ Limitations

- **Under-trained Models**: 1 epoch is a constraint imposed by the assignment; longer training would likely widen the performance gap.
- **Small Hidden Dims**: Hidden dimensions were limited to 16, which is small compared to real-world usage.
- **No Regularization**: Dropout or weight decay was not used, which may have affected generalization.

---

## ✅ Final Claim Verdict

> ✔️ **The data supports the claim**: GeGLU shows consistently higher validation accuracy than ReLU under the given experimental setup.

---

## 🧩 File Overview

| File                             | Purpose                             |
|----------------------------------|-------------------------------------|
| `main_lightning_ver.py`         | Main experiment runner (Lightning)  |
| `*.png` in `data/`               | Saved plots and visualizations      |

---

## ▶️ Running

Install dependencies using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision lightning matplotlib seaborn
python main_lightning_ver.py