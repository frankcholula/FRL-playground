from dataclasses import dataclass


@dataclass
class LunarLanderConfig:
    policy_type: str = "MlpPolicy"
    total_timesteps: int = int(10e5)
    env_name: str = "LunarLander-v3"
