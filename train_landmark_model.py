import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter
import random

# =========================
# SETTINGS
# =========================
DATASET_PATH = "landmark_dataset"
CLASSES = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# =========================
# LOAD DATA
# =========================
X, y = [], []

for idx, label in enumerate(CLASSES):
    file_path = os.path.join(DATASET_PATH, f"{label}.csv")

    if not os.path.exists(file_path):
        continue

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 42:
                X.append([float(i) for i in row])
                y.append(idx)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print("Original Samples:", len(X))

# =========================
# 🔥 BALANCE DATA (IMPORTANT)
# =========================
counts = Counter(y.tolist())
print("Before Balance:", counts)

max_count = max(counts.values())

new_X, new_y = [], []

for cls in counts:
    idxs = [i for i in range(len(y)) if y[i] == cls]

    while len(idxs) < max_count:
        idxs.append(random.choice(idxs))

    for i in idxs:
        new_X.append(X[i].tolist())
        new_y.append(cls)

X = torch.tensor(new_X, dtype=torch.float32)
y = torch.tensor(new_y, dtype=torch.long)

print("After Balance:", Counter(y.tolist()))

# =========================
# SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
class LandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 26)
        )

    def forward(self, x):
        return self.net(x)

model = LandmarkModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# =========================
# TRAIN
# =========================
EPOCHS = 70
best_acc = 0

for epoch in range(EPOCHS):
    model.train()

    # 🔥 AUGMENTATION
    noise = torch.randn_like(X_train) * 0.02
    X_train_aug = X_train + noise

    outputs = model(X_train_aug)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, preds = torch.max(val_outputs, 1)
        acc = (preds == y_val).float().mean()

    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {acc:.2f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "landmark_model.pth")

print("✅ Training complete!")
print(f"Best Accuracy: {best_acc:.2f}")