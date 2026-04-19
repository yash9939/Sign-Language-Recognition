from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from utils.transforms import train_transform, test_transform
import os
from PIL import Image

# Custom Test Dataset
class CustomTestDataset:
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path   # returning path (no label available)


def get_data_loaders(data_dir, batch_size=32):

    # TRAIN (ImageFolder works)
    train_data = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    # Split into train + validation
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # TEST (custom loader)
    test_dataset = CustomTestDataset(
        folder_path=f"{data_dir}/test",
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader