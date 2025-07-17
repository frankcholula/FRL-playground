import numpy as np
import torch
import torch.nn as nn
import minari

import gymnasium as gym
from gymnasium import spaces
from torch.utils.data import DataLoader


# just a simple MLP policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


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


minari_dataset = minari.load_dataset("LunarLanderContinuous-v3/ppo-1000-v1")
dataloader = DataLoader(
    minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
)

env = minari_dataset.recover_environment()

observation_space = env.observation_space
action_space = env.action_space

assert isinstance(observation_space, spaces.Box)
assert isinstance(action_space, spaces.Box)

obs_dim = np.prod(observation_space.shape)
action_dim = action_space.shape[0]
print(f"Observation space dimension: {obs_dim}, Action space dimension: {action_dim}")

policy_net = PolicyNetwork(obs_dim, action_dim)
optimizer = torch.optim.Adam(policy_net.parameters())
loss_fn = nn.MSELoss()

num_epochs = 50
for epoch in range(num_epochs):
    for batch in dataloader:
        observations = batch["observations"][:, :-1]  # Exclude the last observation
        expert_actions = batch["actions"]

        predictioned_actions = policy_net(observations)
        loss = loss_fn(predictioned_actions, expert_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

eval_env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
obs, _ = eval_env.reset()
done = False
accumulated_rew = 0
while not done:
    obs_tensor = torch.Tensor(obs)
    action = policy_net(obs_tensor).detach().numpy()
    obs, rew, ter, tru, _ = eval_env.step(action)
    done = ter or tru
    accumulated_rew += rew

env.close()
print("Accumulated rew: ", accumulated_rew)
