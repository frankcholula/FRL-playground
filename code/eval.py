import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from conf.environment import LunarLanderConfig

model_name = f"ppo-{LunarLanderConfig.env_name}"
eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"))
model = PPO.load(f"models/{model_name}/model", env=eval_env)
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
