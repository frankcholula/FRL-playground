from tqdm.auto import tqdm
from minari import DataCollector
from stable_baselines3 import PPO
import gymnasium as gym
import os

env = DataCollector(gym.make("LunarLander-v3", continuous=True))
path = os.path.abspath("") + "/logs/ppo/LunarLanderContinuous-v3_1/best_model.zip"
agent = PPO.load(path)

total_episodes = 1_000
for i in tqdm(range(total_episodes)):
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = agent.predict(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="LunarLanderContinuous-v3/ppo-1000-v1",
    algorithm_name="ppo",
    code_permalink="https://github.com/frankcholula/FRL-playground/blob/main/code/behavioral_cloning.py",
    author="Frank Lu",
    author_email="lu.phrank@gmail.com",
    description="Behavioral cloning dataset for LunarLanderContinuous-v3 using PPO",
    eval_env="LunarLanderContinuous-v3",
)
