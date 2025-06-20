import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from wandb.integration.sb3 import WandbCallback

env = gym.make("LunarLander-v3", continuous=False, render_mode="human")

observation, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    print("Action taken:", action)

    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(_ + 1))
        observation, info = env.reset()

env.close()
