import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = max(output_dim, input_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)