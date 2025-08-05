import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning as L

# --------- Data ---------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# --------- Models ---------
class FFN_ReLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(784, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, 10) * 0.02)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(torch.einsum('bd,dh->bh', x, self.W_in))
        return torch.einsum('bh,hc->bc', h, self.W_out)

class FFN_GeGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(784, hidden_dim) * 0.02)
        self.W_gate = nn.Parameter(torch.randn(784, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, 10) * 0.02)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_proj = torch.einsum('bd,dh->bh', x, self.W_in)
        x_gate = F.gelu(torch.einsum('bd,dh->bh', x, self.W_gate))
        return torch.einsum('bh,hc->bc', x_proj * x_gate, self.W_out)

# --------- Lightning Wrapper ---------
class LitClassifier(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.accs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.accs.append(acc)

    def on_validation_epoch_end(self):
        avg_acc = torch.stack(self.accs).mean()
        self.log("val_acc", avg_acc, prog_bar=True)
        self.accs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --------- Phase 1: Hidden Dim Sweep ---------
print("\n📊 Phase 1: Hidden Dim Sweep")
hidden_dims = [2, 4, 8, 16]
acc_relu, acc_geglu = [], []

for h in hidden_dims:
    for model_class, accs, name in [(FFN_ReLU, acc_relu, "ReLU"), (FFN_GeGLU, acc_geglu, "GeGLU")]:
        print(f"{name} - Hidden Dim = {h}")
        model = model_class(h)
        lit_model = LitClassifier(model, lr=1e-3)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64)
        trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=False)
        trainer.fit(lit_model, train_loader, val_loader)
        acc = trainer.callback_metrics["val_acc"].item()
        accs.append(acc)
        print(f"  Accuracy = {acc:.4f}")

plt.figure()
plt.plot(hidden_dims, acc_relu, label="ReLU", marker='o')
plt.plot(hidden_dims, acc_geglu, label="GeGLU", marker='o')
plt.xlabel("Hidden Dim")
plt.ylabel("Validation Accuracy")
plt.title("Lightning: Accuracy vs Hidden Dim")
plt.grid(True)
plt.legend()
plt.savefig("lt_hidden_dim_sweep.png")
plt.show()

# --------- Phase 2: k-Trials and Bootstrap ---------
def run_k_trials(model_class, label, k=2):
    print(f"\n🔁 {label}: k={k} Random Trials")
    trials = []
    for i in range(k):
        bs = random.choice([8, 64])
        lr = random.choice([1e-1, 1e-2, 1e-3, 1e-4])
        model = model_class(hidden_dim=8)
        lit_model = LitClassifier(model, lr=lr)
        loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64)
        trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=False)
        trainer.fit(lit_model, loader, val_loader)
        acc = trainer.callback_metrics["val_acc"].item()
        print(f"  Trial {i+1}: BS={bs}  LR={lr:.0e}  Val Acc={acc:.4f}")
        trials.append(acc)
    return trials

def bootstrap_ci(data, samples=10000):
    data = np.array(data)
    assert data.max() <= 1.0, "Accuracy > 1.0 detected. Are you bootstrapping logits?"
    sample_matrix = np.random.choice(data, size=(samples, len(data)), replace=True)
    max_per_sample = sample_matrix.max(axis=1)
    ci = np.percentile(max_per_sample, [2.5, 97.5])
    return max_per_sample, ci

for k in [2, 4, 8]:
    relu_trials = run_k_trials(FFN_ReLU, "ReLU", k)
    geglu_trials = run_k_trials(FFN_GeGLU, "GeGLU", k)

    print("ReLU trials:", relu_trials)
    print("GeGLU trials:", geglu_trials)

    relu_boot, relu_ci = bootstrap_ci(relu_trials)
    geglu_boot, geglu_ci = bootstrap_ci(geglu_trials)

    print(f"ReLU 95% CI: {relu_ci[0]:.4f} – {relu_ci[1]:.4f}")
    print(f"GeGLU 95% CI: {geglu_ci[0]:.4f} – {geglu_ci[1]:.4f}")

    plt.figure()
    sns.histplot(relu_boot, label="ReLU", color="skyblue", bins=20, kde=False)
    sns.histplot(geglu_boot, label="GeGLU", color="orange", bins=20, kde=False)
    plt.axvline(relu_ci[0], linestyle='--', color='blue')
    plt.axvline(relu_ci[1], linestyle='--', color='blue')
    plt.axvline(geglu_ci[0], linestyle='--', color='red')
    plt.axvline(geglu_ci[1], linestyle='--', color='red')
    plt.title(f"Bootstrap Max Validation Accuracy – Lightning (k={k})")
    plt.xlabel("Max Validation Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lt_bootstrap_ci_k{k}.png")
    plt.show()