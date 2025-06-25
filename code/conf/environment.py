from dataclasses import dataclass


@dataclass
class LunarLanderConfig:
    policy_type: str = "MlpPolicy"
    total_timesteps: int = int(1e5)
    env_name: str = "LunarLander-v3"
