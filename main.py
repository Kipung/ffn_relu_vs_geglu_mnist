import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns

# --------- Model Definitions ---------
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

# --------- Data Loading ---------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader   = DataLoader(test_dataset, batch_size=64)

# --------- Train + Evaluate ---------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# --------- Phase 1: Hidden Dim Sweep ---------
print("\n📊 Phase 1: Accuracy vs Hidden Dim")
hidden_dims = [2, 4, 8, 16]
acc_relu = []
acc_geglu = []

for h in hidden_dims:
    print(f"\nHidden Dim = {h}")
    relu = FFN_ReLU(h)
    opt_r = torch.optim.Adam(relu.parameters(), lr=1e-3)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    train_one_epoch(relu, train_loader, opt_r, nn.CrossEntropyLoss())
    acc_r = evaluate(relu, test_loader)
    acc_relu.append(acc_r)
    print(f"FFN_ReLU   Accuracy = {acc_r:.4f}")

    geglu = FFN_GeGLU(h)
    opt_g = torch.optim.Adam(geglu.parameters(), lr=1e-3)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    train_one_epoch(geglu, train_loader, opt_g, nn.CrossEntropyLoss())
    acc_g = evaluate(geglu, test_loader)
    acc_geglu.append(acc_g)
    print(f"FFN_GeGLU  Accuracy = {acc_g:.4f}")

# Plot
plt.plot(hidden_dims, acc_relu, label="ReLU")
plt.plot(hidden_dims, acc_geglu, label="GeGLU")
plt.xlabel("Hidden Dimension")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Hidden Dim")
plt.legend()
plt.grid(True)
plt.show()

# --------- Phase 2: Random Search ---------
print("\n🎯 Phase 2: Random Search (GeGLU only)")
batch_sizes = [8, 64]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
fixed_hidden_dim = 8
search_results = []

for _ in range(10):
    bs = random.choice(batch_sizes)
    lr = random.choice(learning_rates)
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    model = FFN_GeGLU(fixed_hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_one_epoch(model, loader, opt, nn.CrossEntropyLoss())
    acc = evaluate(model, test_loader)
    search_results.append((bs, lr, acc))
    print(f"BS={bs:<3}  LR={lr:<6}  Val Acc = {acc:.4f}")

# --------- Phase 3: V1–V4 Trials ---------
print("\n🔁 Phase 3: V1–V4 Repeats (ReLU & GeGLU)")

def run_trials(model_class, label, runs=4):
    accs = []
    for i in range(runs):
        print(f"{label} - Run {i+1}")
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        model = model_class(fixed_hidden_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_one_epoch(model, loader, opt, nn.CrossEntropyLoss())
        acc = evaluate(model, test_loader)
        accs.append(acc)
        print(f"  Accuracy: {acc:.4f}")
    return accs

relu_trials = run_trials(FFN_ReLU, "ReLU")
geglu_trials = run_trials(FFN_GeGLU, "GeGLU")

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
plt.savefig("bootstrap_histogram.png")
plt.show()