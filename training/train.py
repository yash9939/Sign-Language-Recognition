import torch
import torch.nn as nn
import torch.optim as optim

<<<<<<< HEAD
from models.cnn_model import ASL_CNN
=======
from models.cnn_model import ASL_ResNet 
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253
from utils.dataset_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_data_loaders("Dataset")

<<<<<<< HEAD
model = ASL_CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

EPOCHS = 10
=======
model = ASL_ResNet().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  
optimizer = optim.Adam(model.parameters(), lr=1e-4)   

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

EPOCHS = 20   
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
<<<<<<< HEAD
=======
    correct = 0
    total = 0
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

<<<<<<< HEAD
    # Validation
    model.eval()
    val_loss = 0
=======
        # Accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
<<<<<<< HEAD
            val_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

=======

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    scheduler.step(val_loss)

# Save model
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253
torch.save(model.state_dict(), "saved_models/asl_model.pth")