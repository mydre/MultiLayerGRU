import pdb

import torch
from torch import nn
import torch.nn.functional as F
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print('Mish activation loaded....')

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x