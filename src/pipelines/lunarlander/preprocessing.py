import torch
import numpy as np
from torch.nn.utils import rnn
from torch.utils.data import DataLoader


def collate_fn(batch):
    observations = [torch.as_tensor(x.observations) for x in batch]
    actions = [torch.as_tensor(x.actions) for x in batch]
    rewards = [torch.as_tensor(x.rewards) for x in batch]
    terminations = [torch.as_tensor(x.terminations) for x in batch]
    truncations = [torch.as_tensor(x.truncations) for x in batch]
    episode_lengths = torch.tensor([len(x.actions) for x in batch], dtype=torch.long)
    total_rewards = torch.tensor(
        [np.sum(x.rewards) for x in batch], dtype=torch.float32
    )

    return {
        "id": torch.tensor([x.id for x in batch]),
        "observations": rnn.pad_sequence(observations, batch_first=True),
        "actions": rnn.pad_sequence(actions, batch_first=True),
        "rewards": rnn.pad_sequence(rewards, batch_first=True),
        "terminations": rnn.pad_sequence(terminations, batch_first=True),
        "truncations": rnn.pad_sequence(truncations, batch_first=True),
        "episode_lengths": episode_lengths,
        "total_rewards": total_rewards,
    }


def get_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    all_obs, all_act, all_rew = [], [], []
    for batch in loader:
        all_rew.append(batch["total_rewards"])
        for i in range(batch["observations"].shape[0]):
            length = batch["episode_lengths"][i]
            all_obs.append(batch["observations"][i, :length])
            all_act.append(batch["actions"][i, :length])

    flat_obs = torch.cat(all_obs, dim=0)
    flat_act = torch.cat(all_act, dim=0)
    flat_rew = torch.cat(all_rew, dim=0)

    stats = {
        "obs_mean": torch.mean(flat_obs, dim=0),
        "obs_std": torch.std(flat_obs, dim=0),
        "act_mean": torch.mean(flat_act, dim=0),
        "act_std": torch.std(flat_act, dim=0),
        "rew_mean": torch.mean(flat_rew),
        "rew_std": torch.std(flat_rew),
    }
    stats["obs_std"][stats["obs_std"] < 1e-6] = 1e-6
    stats["act_std"][stats["act_std"] < 1e-6] = 1e-6
    stats["rew_std"] = 1e-6 if stats["rew_std"] < 1e-6 else stats["rew_std"]
    return stats


def create_trajectory_chunks(batch, horizon):
    batch_size = batch["observations"].shape[0]
    all_chunks = []

    for i in range(batch_size):
        # Get the data for one episode and its true length
        obs = batch["observations"][i]
        act = batch["actions"][i]
        length = batch["episode_lengths"][i]

        # Slide a window of size 'horizon' over the valid part of the episode
        for start_idx in range(length - horizon + 1):
            end_idx = start_idx + horizon

            obs_chunk = obs[start_idx:end_idx]
            act_chunk = act[start_idx:end_idx]
            chunk = torch.cat([obs_chunk, act_chunk], dim=-1)
            all_chunks.append(chunk.flatten())

    if not all_chunks:
        return None

    return torch.stack(all_chunks)


def create_normalized_chunks(batch, horizon, stats, cond_type=None):
    """
    Creates normalized chunks. Can be unconditional, or conditional on
    total reward or the starting observation of the chunk.

    Args:
        cond_type (str, optional): Can be 'reward' or 'start_obs'. Defaults to None.
    """
    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    act_mean, act_std = stats["act_mean"], stats["act_std"]

    all_chunks = []
    all_conds = [] if cond_type else None

    for i in range(batch["observations"].shape[0]):
        obs, act, length = (
            batch["observations"][i],
            batch["actions"][i],
            batch["episode_lengths"][i],
        )

        if length < horizon:
            continue

        # If conditioning on reward, calculate it once per episode
        if cond_type == "reward":
            rew_mean, rew_std = stats["rew_mean"], stats["rew_std"]
            total_reward = batch["total_rewards"][i]
            norm_cond = (total_reward - rew_mean) / rew_std

        # Slide the window across the episode
        for start_idx in range(length - horizon + 1):
            end_idx = start_idx + horizon
            obs_chunk = obs[start_idx:end_idx]
            act_chunk = act[start_idx:end_idx]

            # Normalize the main trajectory chunk
            norm_obs_chunk = (obs_chunk - obs_mean) / obs_std
            norm_act_chunk = (act_chunk - act_mean) / act_std
            chunk = torch.cat([norm_obs_chunk, norm_act_chunk], dim=-1)
            all_chunks.append(chunk.flatten())

            # If conditional, prepare the corresponding condition
            if cond_type == "start_obs":
                start_obs = obs_chunk[0]
                norm_cond = (start_obs - obs_mean) / obs_std
                all_conds.append(norm_cond)
            elif cond_type == "reward":
                all_conds.append(norm_cond)

    if not all_chunks:
        return (None, None) if cond_type else None

    stacked_chunks = torch.stack(all_chunks)

    if cond_type:
        stacked_conds = torch.stack(all_conds)
        if cond_type == "reward":
            stacked_conds = stacked_conds.unsqueeze(1)
        return stacked_chunks, stacked_conds
    else:
        return stacked_chunks


def unnormalize_trajectory(chunk, stats, horizon, obs_dim, action_dim):
    obs_mean, obs_std = stats["obs_mean"].to(chunk.device), stats["obs_std"].to(
        chunk.device
    )
    act_mean, act_std = stats["act_mean"].to(chunk.device), stats["act_std"].to(
        chunk.device
    )
    reshaped = chunk.reshape(horizon, obs_dim + action_dim)
    norm_obs = reshaped[:, :obs_dim]
    norm_act = reshaped[:, obs_dim:]
    obs = norm_obs * obs_std + obs_mean
    act = norm_act * act_std + act_mean
    return obs, act
