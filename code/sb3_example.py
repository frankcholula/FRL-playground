import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
import os
from dotenv import load_dotenv
import glob

# --- Configuration ---
load_dotenv()
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
model_name = f"ppo-{config['env_name']}"
model_dir = os.path.join(ROOT_DIR, "models")
video_dir = os.path.join(ROOT_DIR, "videos")
runs_dir = os.path.join(ROOT_DIR, "runs")

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(runs_dir, exist_ok=True)


# --- W&B Initialization ---
run = wandb.init(
    project="sb3-video-fix-demo",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,       # still useful for capturing other stats
    save_code=True,         # optional
)

# --- Custom Callback for Video Logging ---
class VideoRecorderCallback(BaseCallback):
    """
    Callback for rendering a video of the agent interacting with the environment
    and logging it to W&B.
    
    :param video_folder: Path to the folder where videos are saved.
    :param video_length: Length of the recorded video.
    """
    def __init__(self, video_folder: str, verbose: int = 0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.logged_videos = set() # Keep track of logged videos to avoid duplicates

    def _on_step(self) -> bool:
        # Check for new video files in the designated folder
        video_files = glob.glob(os.path.join(self.video_folder, "*.mp4"))
        
        for video_file in video_files:
            if video_file not in self.logged_videos:
                self.logger.info(f"Found new video to log: {video_file}")
                
                # Log the video to W&B
                wandb.log({
                    "video": wandb.Video(video_file, caption=f"Step {self.num_timesteps}", fps=20)
                })
                
                # Add to the set of logged videos
                self.logged_videos.add(video_file)
                
        return True


# --- Environment Setup ---
def make_env():
    """
    Utility function for multiprocessed env.
    """
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = Monitor(env)  # record stats such as returns
    return env

# The environment must be wrapped in a DummyVecEnv for the VecVideoRecorder
env = DummyVecEnv([make_env])

# This is the folder where the videos will be saved
# We use the W&B run id to have a unique folder for each run
run_video_folder = f"{video_dir}/{run.id}"

env = VecVideoRecorder(
    env,
    run_video_folder,
    record_video_trigger=lambda x: x % 2000 == 0, # Records a video every 2000 steps
    video_length=200,
)

# --- Model Training ---
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"{runs_dir}/{run.id}")

# Create a callback list with both W&B and our custom video logger
callback_list = CallbackList([
    WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"{model_dir}/{run.id}",
        verbose=2,
    ),
    VideoRecorderCallback(run_video_folder)
])

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callback_list,
)

run.finish()
print("Training finished and run closed.")
