import torch
from torch import nn, Tensor


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x
