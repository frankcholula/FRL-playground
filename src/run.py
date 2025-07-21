from src.utils.args import parse_args
import torch


def main():
    args = parse_args()
    torch.manual_seed(42)
    print(f"Arguments: {args}")


if __name__ == "__main__":
    main()
