'''
@Author:        ZM
@Date and Time: 2020/6/7 9:37
@File:          PFLU.py
'''

import torch
from torch import nn
from torch.autograd import Function

'''
https://doi.org/10.1016/j.neucom.2020.11.068
PFLU and FPFLU: Two novel non-monotonic activation functions in convolutional neural networks
'''
class PFLUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x, )

        return x * (1 + x / torch.sqrt(1 + x * x)) / 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            t = 1 / (1 + x * x)
            grad_x = grad_output * (1 + x * torch.sqrt(t) * (1 + t)) / 2

        return grad_x

class MemoryEfficientPFLU(nn.Module):
    def forward(self, x):
        return PFLUFunction.apply(x)

class PFLU(nn.Module):
    def forward(self, x):
        return x * (1 + x / torch.sqrt(1 + x * x)) / 2
