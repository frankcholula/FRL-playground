import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from wandb.integration.sb3 import WandbCallback

from conf.environment import LunarLanderConfig
from dotenv import load_dotenv
import wandb

load_dotenv()
model_name = f"ppo-{LunarLanderConfig.env_name}"


def make_env():
    env = gym.make(
        LunarLanderConfig.env_name, continuous=False, render_mode="rgb_array"
    )
    return env


run = wandb.init(
    project="FRL",
    config=LunarLanderConfig,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=False,
)
env = make_vec_env(make_env, n_envs=16)
env = VecVideoRecorder(
    env,
    f"videos/{model_name}",
    record_video_trigger=lambda x: x % 4000 == 0,
    video_length=250,
)
model = PPO(
    policy=LunarLanderConfig.policy_type,
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="runs",
)
model.learn(
    total_timesteps=LunarLanderConfig.total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{model_name}",
        verbose=2,
    ),
)
run.finish()
