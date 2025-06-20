import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from wandb.integration.sb3 import WandbCallback
import os
import wandb

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": int(1e5),
    "env_name": "LunarLander-v3",
}

model_name = f"ppo-{config['env_name']}"
model_dir = os.path.join(ROOT_DIR, "../models")
video_dir = os.path.join(ROOT_DIR, "../videos")


def make_env():
    return gym.make(config["env_name"], continuous=False, render_mode="rgb_array")


if model_name not in os.listdir(model_dir):
    run = wandb.init(
        project="FRL",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=False,
    )
    env = make_vec_env(make_env, n_envs=16)
    env = VecVideoRecorder(
        env,
        f"{video_dir}/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=200,
    )
    model = PPO(
        policy=config["policy_type"],
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"{model_dir}/{run.id}",
            verbose=2,
        ),
    )
    run.finish()

else:
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"))
    model = PPO.load(model_name, env=eval_env)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
