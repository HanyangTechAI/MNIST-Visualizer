import torch
from torch import nn

from model.module import ChannelAttention, SpatialAttention

class ResBlock(nn.Module):
    def __init__(self, channel: int, ratio=16):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv_upper = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel)
        )

        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        path = self.ca(path) * path
        path = self.sa(path) * path

        return self.relu(path + x)

class Network(nn.Module):
    def __init__(self, in_channel: int, filters:int, blocks:int, num_classes: int):
        super(Network, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),

            *[ResBlock(filters) for _ in range(blocks)],
            
            nn.Conv2d(filters, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 ** 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)

        return self.classifier(x)

    def predict_with_feature(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.feature_extractor(x)

            return self.classifier(x), x
