import torch.nn as nn
import torch
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self, label_num=10, channels=3, height=32, width=32):
        super(Generator, self).__init__()
        self.x, self.y = int(height / 4), int(width / 4)
        self.k1, self.k2 = height % 4, width % 4
        if self.k1 == 0:
            self.k1, self.x = 4, self.x - 1
        if self.k2 == 0:
            self.k2, self.y = 4, self.y - 1
        self.linear1 = nn.Linear(label_num, 128 * self.x * self.y)
        self.bn1 = nn.BatchNorm1d(128 * self.x * self.y)
        self.linear2 = nn.Linear(1000, 128 * self.x * self.y)
        self.bn2 = nn.BatchNorm1d(128 * self.x * self.y)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), padding=1),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2,padding=1),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, kernel_size=(self.k1, self.k2), stride=2,padding=1),
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x1, x2):
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x1 = x1.view(-1, 128, self.x, self.y)
        x2 = F.relu(self.linear2(x2))
        x2 = self.bn2(x2)
        x2 = x2.view(-1, 128, self.x, self.y)
        x = torch.cat([x1, x2], axis=1)
        x = self.deconv1(x)
        return x