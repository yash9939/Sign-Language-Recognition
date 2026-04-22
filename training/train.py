import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from models.cnn_model import ASL_ResNet
from utils.dataset_loader import get_data_loaders

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
train_loader, val_loader, _, class_names = get_data_loaders("Dataset")

# =========================
# MODEL
# =========================
model = ASL_ResNet(num_classes=len(class_names)).to(device)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3
)

# =========================
# TRAINING
# =========================
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    scheduler.step(val_loss)

# =========================
# SAVE MODEL
# =========================
os.makedirs("saved_models", exist_ok=True)

torch.save(model.state_dict(), "saved_models/asl_model.pth")

# Save class names (VERY IMPORTANT)
with open("saved_models/classes.json", "w") as f:
    json.dump(class_names, f)

print("✅ Model and classes saved successfully!")