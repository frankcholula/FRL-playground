from src.utils.args import parse_args
import torch
from models.cnn import TemporalCNN
from models.mlp import MLP

# Data preprocessing functions
def main():
    args = parse_args()
    torch.manual_seed(42)
    print(f"Arguments: {args}")


if __name__ == "__main__":
    main()
