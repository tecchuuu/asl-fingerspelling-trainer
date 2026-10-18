from torch import nn

from src import config


class MLP(nn.Module):
    """Model B: 63 normalized landmark features -> 24 letter scores."""

    def __init__(self, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        in_size = config.NUM_LANDMARKS * 3
        for size in hidden:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, config.NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Raw scores (logits); CrossEntropyLoss applies softmax itself
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
