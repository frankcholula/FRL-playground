import argparse
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow matching trajectory generation.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="LunarLanderContinuous-v3/ppo-1000-deterministic-v1",
        help="Name of the dataset to use.",
    )
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-type", type=str, default="cnn", choices=["mlp", "cnn"])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()
