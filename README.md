# ReLU vs GeGLU on MNIST — Feedforward MLP Comparison

This repo contains experiments comparing **FFN_ReLU** and **FFN_GeGLU** models on the MNIST dataset using both standard PyTorch and PyTorch Lightning.

## 🧪 Experiment Pipeline

### 1. Hidden Dimension Sweep
- Tested hidden_dims = `[2, 4, 8, 16]`
- Measured test accuracy per model

### 2. Random Search
- Hyperparameter space:
  - Batch Size: `[8, 64]`
  - Learning Rate: `[1e-1, 1e-2, 1e-3, 1e-4]`
- 10 random trials to pick best settings

### 3. Multiple Runs (V1–V4)
- Ran each model 4 times with best (BS, LR)
- Collected max validation accuracies

### 4. Bootstrap Sampling
- Used 10,000 samples with replacement from V1–V4
- Computed 95% confidence interval for each model
- Compared accuracy distributions

## 📄 Files

| File                  | Description                            |
|-----------------------|----------------------------------------|
| `main_non_lightning.py` | PyTorch implementation                |
| `main_lightning.py`     | PyTorch Lightning version             |
| `bootstrap_plot.png`    | Final comparison plot of both models |
| `requirements.txt`      | Minimal dependencies                  |

## 🔧 Install Dependencies

```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
