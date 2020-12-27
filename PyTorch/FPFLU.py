'''
@Author:        ZM
@Date and Time: 2020/6/7 9:38
@File:          FPFLU.py
'''

import torch
from torch import nn
from torch.autograd import Function

class FPFLUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        return torch.maximum(x, x / (1 + x * x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            u = x * x
            v = 1 + u
            one_tensor = torch.ones(1, dtype=x.dtype, device=x.device)
            grad_x = grad_output * torch.where(x > 0, one_tensor, (1 - u) / (v * v))
        return grad_x

class MemoryEfficientFPFLU(nn.Module):
    def forward(self, x):
        return FPFLUFunction.apply(x)

class FPFLU(nn.Module):
    def forward(self, x):
        return torch.maximum(x, x / (1 + x * x))