import torch
from torch import nn, Tensor
from models.activation_fns import Swish


class MLP(nn.Module):
    def __init__(self, input_dim: int, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)


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


# TODO: Implement UNet
class UNet(nn.Module):
    pass

# TODO: Implement ControlNet for conditioning case.
class ControlNet(nn.Module):
    pass