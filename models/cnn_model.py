import torch.nn as nn


class ASL_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            self.block(3,32),
            self.block(32,64),
            self.block(64,128),
            self.block(128,256)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,29)
        )

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

from torchvision import models

class ASL_ResNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )## Forward pass simply goes through the backbone

    def forward(self, x):
        return self.backbone(x)
