import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = max(output_dim, input_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Initialize weights with smaller values for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)