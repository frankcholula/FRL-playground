from flow_matching import nn as fmnn

import torch
from torch import nn, Tensor

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper


# pi(a) -> a
# (t, state, action) -> a


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x
    
class ConditionalVectorField(nn.Module):
    def __init__(self, )


