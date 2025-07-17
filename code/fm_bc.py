import time
import torch

from torch import nn, Tensor
from torch.utils.data import DataLoader

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt

# dataset
import minari

# WandB
from utils.loggers import WandBLogger

# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using gpu")
else:
    device = "cpu"
    print("Using cpu.")
torch.manual_seed(42)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# TODO: need to resolve temporal locality problem maybe with a CNN later.
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


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch], batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch], batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


minari_dataset = minari.load_dataset(dataset_id="LunarLanderContinuous-v3/ppo-1000-v1")
dataloader = DataLoader(
    minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
)
env = minari_dataset.recover_environment()
episode = minari_dataset[0]


# Training
horizon = 100
action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
input_dim = (obs_dim + action_dim) * horizon
print(
    f"Input dimension: {input_dim} made of {obs_dim} observations and {action_dim} actions over {horizon} timesteps."
)
# Training params
lr = 0.001
num_epochs = 1000
print_every = 10
hidden_dim = 512
print(
    f"Training with {num_epochs} epochs, {print_every} print frequency, and {hidden_dim} hidden dimension."
)

vf = MLP(input_dim=input_dim, time_dim=1, hidden_dim=hidden_dim).to(device)
path = AffineProbPath(scheduler=CondOTScheduler())
optim = torch.optim.Adam(vf.parameters(), lr=lr)


config = {
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "hidden_dim": hidden_dim,
    "horizon": horizon,
    "batch_size": dataloader.batch_size,
    "input_dim": input_dim,
    "obs_dim": obs_dim,
    "action_dim": action_dim,
}
logger = WandBLogger(config=config)

print("Starting training...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    start_time = time.time()

    for batch in dataloader:
        optim.zero_grad()

        observations = batch["observations"][:, :-1][:, :horizon]
        expert_actions = batch["actions"][:, :horizon]
        x_1 = torch.cat([observations, expert_actions], dim=-1)
        x_1 = x_1.reshape(x_1.shape[0], -1).to(device)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(x_1.shape[0]).to(device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        predicted_velocity = vf(path_sample.x_t, path_sample.t)
        loss = torch.pow(predicted_velocity - path_sample.dx_t, 2).mean()

        logger.log({"batch_loss": loss.item()})

        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    logger.log({"avg_epoch_loss": avg_epoch_loss})

    if (epoch + 1) % print_every == 0:
        elapsed = time.time() - start_time
        print(
            f"| Epoch {epoch+1:6d} | {elapsed:.2f} s/epoch | Loss {avg_epoch_loss:8.5f} "
        )
        start_time = time.time()
print("Training finished.")
