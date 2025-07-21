import torch
from torch import nn, Tensor
from activation_fns import Swish


class TemporalCNN(nn.Module):
    def __init__(self, horizon: int, transition_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        input_channels = transition_dim + 1
        self.main = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding="same"),
            Swish(),
            nn.Conv1d(hidden_dim, transition_dim, kernel_size=5, padding="same"),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_reshaped = x.view(-1, self.horizon, self.transition_dim).permute(0, 2, 1)
        t_expanded = t.view(-1, 1, 1).expand(-1, 1, self.horizon)
        h = torch.cat([x_reshaped, t_expanded], dim=1)
        out = self.main(h)
        return out.permute(0, 2, 1).reshape(x.shape)
