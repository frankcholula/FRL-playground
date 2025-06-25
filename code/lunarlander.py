import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

from conf.environment import LunarLanderConfig
from dotenv import load_dotenv
import wandb

load_dotenv()


def make_env():
    return gym.make(
        LunarLanderConfig.env_name, continuous=False, render_mode="rgb_array"
    )


model_name = f"ppo-{LunarLanderConfig.env_name}"


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
    f"videos/{run.id}",
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
    tensorboard_log=f"runs/{run.id}",
)
model.learn(
    total_timesteps=LunarLanderConfig.total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
        log="all",
    ),
)
run.finish()

# eval_env = Monitor(gym.make("LunarLander-v3", render_mode="human"))
# model = PPO.load(model_name, env=eval_env)
# mean_reward, std_reward = evaluate_policy(
#     model, eval_env, n_eval_episodes=10, deterministic=True
# )
# print(f"Mean reward: {mean_reward} +/- {std_reward}")
