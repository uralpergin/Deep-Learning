import torch
import torch.nn as nn


class SimpleDiffusion(nn.Module):
    def __init__(self, n_features: int, n_blocks: int = 2, n_units: int = 64):
        super(SimpleDiffusion, self).__init__()

        self.linear_projection = torch.nn.Linear(
            in_features=n_features + 1, out_features=n_units
        )
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(n_units, n_units) for i in range(n_blocks)]
        )
        self.relu = torch.nn.ReLU()
        self.outblock = nn.Linear(n_units, n_features)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_cat = torch.hstack([x, t])
        input_projection = self.linear_projection(input_cat)
        for layer in self.linear_layers:
            input_projection = self.relu(layer(input_projection))
        output = self.outblock(input_projection)
        return output
