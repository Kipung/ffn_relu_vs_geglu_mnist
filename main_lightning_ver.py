import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
        self.net = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class FFN_GeGLU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(784, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_proj, x_gate = self.proj(x).chunk(2, dim=-1)
        return self.out(x_proj * F.gelu(x_gate))

# --------- Lightning Module ---------
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
    for model_type, accs in [("ReLU", acc_relu), ("GeGLU", acc_geglu)]:
        print(f"{model_type} - Hidden Dim = {h}")
        model = FFN_ReLU(h) if model_type == "ReLU" else FFN_GeGLU(h)
        lit_model = LitClassifier(model, lr=1e-3)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64)
        trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=True)
        trainer.fit(lit_model, train_loader, val_loader)
        acc = trainer.callback_metrics["val_acc"].item()
        accs.append(acc)
        print(f"  Accuracy = {acc:.4f}")

# Plot
plt.plot(hidden_dims, acc_relu, label="ReLU")
plt.plot(hidden_dims, acc_geglu, label="GeGLU")
plt.xlabel("Hidden Dim")
plt.ylabel("Validation Accuracy")
plt.title("Lightning: Accuracy vs Hidden Dim")
plt.grid(True)
plt.legend()
plt.show()

# --------- Phase 2: Random Search (GeGLU) ---------
print("\n🔍 Phase 2: Random Search (GeGLU)")
batch_sizes = [8, 64]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
fixed_hidden_dim = 8

for _ in range(10):
    bs = random.choice(batch_sizes)
    lr = random.choice(learning_rates)
    model = FFN_GeGLU(fixed_hidden_dim)
    lit_model = LitClassifier(model, lr)
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64)
    trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=True)
    trainer.fit(lit_model, loader, val_loader)
    acc = trainer.callback_metrics["val_acc"].item()
    print(f"BS={bs:<3}  LR={lr:<6}  Val Acc = {acc:.4f}")

# --------- Phase 3: V1–V4 Trials ---------
def run_v_trials(model_class, label, runs=4):
    print(f"\n🔁 Phase 3: V1–V4 Trials for {label}")
    accs = []
    for i in range(runs):
        print(f"{label} - Run {i+1}")
        model = model_class(fixed_hidden_dim)
        lit_model = LitClassifier(model, lr=1e-3)
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=64)
        trainer = L.Trainer(max_epochs=1, logger=False, enable_progress_bar=True)
        trainer.fit(lit_model, loader, val_loader)
        acc = trainer.callback_metrics["val_acc"].item()
        accs.append(acc)
        print(f"  Accuracy = {acc:.4f}")
    return accs

relu_trials = run_v_trials(FFN_ReLU, "ReLU")
geglu_trials = run_v_trials(FFN_GeGLU, "GeGLU")

# --------- Phase 4: Bootstrap Sampling ---------
print("\n📈 Phase 4: Bootstrap Confidence Intervals")

def bootstrap_ci(data, samples=10000):
    sample_matrix = np.random.choice(data, size=(samples, len(data)), replace=True)
    max_per_sample = sample_matrix.max(axis=1)
    ci = np.percentile(max_per_sample, [2.5, 97.5])
    return max_per_sample, ci

relu_boot, relu_ci = bootstrap_ci(relu_trials)
geglu_boot, geglu_ci = bootstrap_ci(geglu_trials)

print(f"ReLU  CI 95%: {relu_ci[0]:.4f} – {relu_ci[1]:.4f}")
print(f"GeGLU CI 95%: {geglu_ci[0]:.4f} – {geglu_ci[1]:.4f}")

# Plot
plt.figure(figsize=(10, 5))
sns.histplot(relu_boot, label="ReLU", kde=True, color="skyblue")
sns.histplot(geglu_boot, label="GeGLU", kde=True, color="orange")
plt.axvline(relu_ci[0], linestyle='--', color='blue')
plt.axvline(relu_ci[1], linestyle='--', color='blue')
plt.axvline(geglu_ci[0], linestyle='--', color='red')
plt.axvline(geglu_ci[1], linestyle='--', color='red')
plt.title("Bootstrap Max Accuracy CI (10,000 Samples)")
plt.xlabel("Max Accuracy")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("bootstrap_lightning_histogram.png")
plt.show()