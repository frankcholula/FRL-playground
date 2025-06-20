import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from wandb.integration.sb3 import WandbCallback
import os

env_name = "LunarLander-v3"
model_name = f"ppo-{env_name}"
models_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../models")

if model_name not in os.listdir("models_dir"):
    env = make_vec_env(lambda: gym.make(env_name, continuous=False), n_envs=16)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=int(10e5))
    model.save(os.path.join("models", model_name))

else:
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"))
    model = PPO.load(model_name, env=eval_env)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
