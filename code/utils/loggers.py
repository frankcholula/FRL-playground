from stable_baselines3.common.callbacks import BaseCallback
import wandb
import os


class VideoLoggingCallback(BaseCallback):
    def __init__(self, video_dir, check_freq=2000, verbose=0):
        super().__init__(verbose)
        self.video_dir = video_dir
        self.check_freq = check_freq
        self.logged_files = set()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            if os.path.exists(self.video_dir):
                for fname in os.listdir(self.video_dir):
                    if fname.endswith(".mp4") and fname not in self.logged_files:
                        fpath = os.path.join(self.video_dir, fname)
                        wandb.log({f"video": wandb.Video(fpath, format="mp4")})
                        self.logged_files.add(fname)
                        if self.verbose > 0:
                            print(
                                f"[W&B] Logged video: {fname} at step {self.num_timesteps}"
                            )
        return True
