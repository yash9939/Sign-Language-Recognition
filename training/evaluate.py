import torch
<<<<<<< HEAD
from models.cnn_model import ASL_CNN
from utils.dataset_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ASL_CNN().to(device)
model.load_state_dict(torch.load("saved_models/asl_model.pth"))

_, _, test_loader = get_data_loaders("Dataset")
=======
from models.cnn_model import ASL_ResNet   # ✅ if using ResNet
from utils.dataset_loader import get_data_loaders
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ASL_ResNet().to(device)
model.load_state_dict(torch.load("saved_models/asl_model.pth", map_location=device))

# Load class names (IMPORTANT) ... to ensure correct mapping
with open("saved_models/classes.json", "r") as f:
    classes = json.load(f)

_, _, test_loader, _ = get_data_loaders("Dataset")
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253

model.eval()

with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        for i in range(len(paths)):
<<<<<<< HEAD
            print(f"{paths[i]} → Predicted: {preds[i].item()}")
=======
            pred_label = classes[preds[i].item()]
            print(f"{paths[i]} → Predicted: {pred_label}")
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253
