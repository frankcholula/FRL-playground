import argparse


def get_args() -> argparse.Namespace:
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
    return parser.parse_args()
