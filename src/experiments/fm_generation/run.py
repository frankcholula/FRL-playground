import time
import torch
import random
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

import minari

from src.utils.loggers import WandBLogger
from .args import get_args


def collate_fn(batch):
    observations = [torch.as_tensor(x.observations) for x in batch]
    actions = [torch.as_tensor(x.actions) for x in batch]
    rewards = [torch.as_tensor(x.rewards) for x in batch]
    terminations = [torch.as_tensor(x.terminations) for x in batch]
    truncations = [torch.as_tensor(x.truncations) for x in batch]
    episode_lengths = torch.tensor([len(x.actions) for x in batch], dtype=torch.long)

    return {
        "id": torch.tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(observations, batch_first=True),
        "actions": torch.nn.utils.rnn.pad_sequence(actions, batch_first=True),
        "rewards": torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True),
        "terminations": torch.nn.utils.rnn.pad_sequence(terminations, batch_first=True),
        "truncations": torch.nn.utils.rnn.pad_sequence(truncations, batch_first=True),
        "episode_lengths": episode_lengths,
    }


def get_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    all_obs, all_act = [], []
    for batch in loader:
        for i in range(batch["observations"].shape[0]):
            length = batch["episode_lengths"][i]
            all_obs.append(batch["observations"][i, :length])
            all_act.append(batch["actions"][i, :length])

    flat_obs = torch.cat(all_obs, dim=0)
    flat_act = torch.cat(all_act, dim=0)
    stats = {
        "obs_mean": torch.mean(flat_obs, dim=0),
        "obs_std": torch.std(flat_obs, dim=0),
        "act_mean": torch.mean(flat_act, dim=0),
        "act_std": torch.std(flat_act, dim=0),
    }
    stats["obs_std"][stats["obs_std"] < 1e-6] = 1e-6
    stats["act_std"][stats["act_std"] < 1e-6] = 1e-6
    return stats


def create_normalized_chunks(batch, horizon, stats):
    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    act_mean, act_std = stats["act_mean"], stats["act_std"]

    all_chunks = []
    for i in range(batch["observations"].shape[0]):
        obs, act, length = (
            batch["observations"][i],
            batch["actions"][i],
            batch["episode_lengths"][i],
        )
        if length < horizon:
            continue
        for start_idx in range(length - horizon + 1):
            end_idx = start_idx + horizon
            obs_chunk = obs[start_idx:end_idx]
            act_chunk = act[start_idx:end_idx]
            norm_obs_chunk = (obs_chunk - obs_mean) / obs_std
            norm_act_chunk = (act_chunk - act_mean) / act_std
            chunk = torch.cat([norm_obs_chunk, norm_act_chunk], dim=-1)
            all_chunks.append(chunk.flatten())
    if not all_chunks:
        return None
    return torch.stack(all_chunks)


def unnormalize_trajectory(chunk, stats, horizon, obs_dim, action_dim):
    obs_mean, obs_std = stats["obs_mean"].to(chunk.device), stats["obs_std"].to(chunk.device)
    act_mean, act_std = stats["act_mean"].to(chunk.device), stats["act_std"].to(chunk.device)
    reshaped = chunk.reshape(horizon, obs_dim + action_dim)
    norm_obs = reshaped[:, :obs_dim]
    norm_act = reshaped[:, obs_dim:]
    obs = norm_obs * obs_std + obs_mean
    act = norm_act * act_std + act_mean
    return obs, act


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


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


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


def train(args):
    dataset = minari.load_dataset(dataset_id=args.dataset_name)
    env = dataset.recover_environment()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    transition_dim = obs_dim + action_dim
    input_dim = args.horizon * transition_dim

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    stats = get_dataset_stats(dataset)

    model = TemporalCNN(horizon=args.horizon, transition_dim=transition_dim, hidden_dim=args.hidden_dim).to(args.device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger = None
    if not args.no_wandb:
        logger = WandBLogger(
            config={
                "horizon": args.horizon,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
            }
        )

    print("Starting training...")
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        total_chunks = 0
        start_time = time.time()
        for batch in dataloader:
            optim.zero_grad()
            x1 = create_normalized_chunks(batch, args.horizon, stats)
            if x1 is None:
                continue
            x1 = x1.to(args.device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=args.device)
            sample = path.sample(t=t, x_0=x0, x_1=x1)
            pred = model(sample.x_t, sample.t)
            loss = ((pred - sample.dx_t) ** 2).mean()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_chunks += 1
        avg_loss = total_loss / total_chunks if total_chunks > 0 else 0.0
        if logger:
            logger.log({"avg_epoch_loss": avg_loss})
        if (epoch + 1) % args.print_every == 0:
            elapsed = time.time() - start_time
            print(f"| Epoch {epoch+1:6d} | {elapsed:.2f} s/epoch | Loss {avg_loss:8.5f} |")
            start_time = time.time()
    if logger:
        logger.finish()
    return model, stats, input_dim, obs_dim, action_dim


def generate(model, stats, input_dim, obs_dim, action_dim, args):
    wrapped = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped)
    T = torch.linspace(0, 1, 10, device=args.device)
    x_init = torch.randn((1, input_dim), device=args.device)
    sol = solver.sample(time_grid=T, x_init=x_init, method="midpoint", step_size=0.05)
    final_chunk = sol[-1].squeeze(0).detach()
    return unnormalize_trajectory(final_chunk, stats, args.horizon, obs_dim, action_dim)


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device:
        torch.set_default_device(args.device)
    model, stats, input_dim, obs_dim, action_dim = train(args)
    obs, act = generate(model, stats, input_dim, obs_dim, action_dim, args)
    print("Generated observation shape:", obs.shape)
    print("Generated action shape:", act.shape)


if __name__ == "__main__":
    main()
