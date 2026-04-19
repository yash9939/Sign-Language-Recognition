import torch
from models.cnn_model import ASL_CNN
from utils.dataset_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ASL_CNN().to(device)
model.load_state_dict(torch.load("models/asl_model.pth"))

_, _, test_loader = get_data_loaders("Dataset")

model.eval()

with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        for i in range(len(paths)):
            print(f"{paths[i]} → Predicted: {preds[i].item()}")