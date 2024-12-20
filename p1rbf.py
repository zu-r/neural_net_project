import torch
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()
        self.param_vecs = nn.Parameter(torch.tensor([
            ParamVecs.ZERO,
            ParamVecs.ONE,
            ParamVecs.TWO,
            ParamVecs.THREE,
            ParamVecs.FOUR,
            ParamVecs.FIVE,
            ParamVecs.SIX,
            ParamVecs.SEVEN,
            ParamVecs.EIGHT,
            ParamVecs.NINE
        ], dtype=torch.float32))
        self.param_vecs.requires_grad = False

    def forward(self, x):
        x_expanded = x.unsqueeze(1).expand(-1, 10, -1)
        return torch.norm(x_expanded - self.param_vecs, dim=2)

class ParamVecs:
    ZERO = [
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1, -1, -1, -1,  1,  1,
         1, -1, -1,  1, -1, -1,  1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
         1, -1, -1,  1, -1, -1,  1,
         1,  1, -1, -1, -1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    ONE = [
         1,  1,  1, -1, -1,  1,  1,
         1,  1, -1, -1, -1,  1,  1,
         1, -1, -1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    TWO = [
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1, -1, -1, -1, -1, -1,  1,
        -1, -1,  1,  1,  1, -1, -1,
        -1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1, -1, -1,  1,
         1,  1, -1, -1, -1,  1,  1,
         1, -1, -1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    THREE = [
        -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1, -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
         1, -1, -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    FOUR = [
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1, -1, -1,  1,  1, -1, -1,
         1, -1, -1,  1,  1, -1, -1,
        -1, -1, -1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1,  1, -1, -1
    ]
    FIVE = [
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,  1,  1,
         1,  1, -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
         1, -1, -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    SIX = [
         1,  1, -1, -1, -1, -1,  1,
         1, -1, -1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,
        -1, -1, -1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1, -1,  1,  1, -1, -1,
         1, -1, -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    SEVEN = [
        -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,
         1,  1, -1, -1,  1,  1,  1,
         1,  1, -1, -1,  1,  1,  1,
         1,  1, -1, -1,  1,  1,  1,
         1,  1, -1, -1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    EIGHT = [
         1, -1, -1, -1, -1, -1,  1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
         1, -1, -1, -1, -1, -1,  1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
         1, -1, -1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]
    NINE = [
         1, -1, -1, -1, -1, -1,  1,
        -1, -1,  1,  1, -1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1, -1, -1, -1,
         1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1,  1, -1, -1,
         1,  1,  1,  1, -1, -1,  1,
         1, -1, -1, -1, -1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1
    ]