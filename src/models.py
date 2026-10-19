from torch import nn

from src import config


class MLP(nn.Module):
    """Model B: 63 normalized landmark features -> 24 letter scores (25 with the "other" class)."""

    def __init__(self, hidden=(256, 128), dropout=0.3, num_classes=config.NUM_CLASSES):
        super().__init__()
        layers = []
        in_size = config.NUM_LANDMARKS * 3
        for size in hidden:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Raw scores (logits); CrossEntropyLoss applies softmax itself
        return self.net(x)


class CNN(nn.Module):
    """Model A (baseline): 64x64 RGB hand crop -> 24 letter scores.

    Three conv blocks, each halving the image and doubling the channels:
    3x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8, then average each channel to one number.
    """

    def __init__(self, channels=(32, 64, 128), dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
