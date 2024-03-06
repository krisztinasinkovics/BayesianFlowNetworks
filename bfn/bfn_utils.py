import torch
from torch import Tensor
import numpy as np
import math

import matplotlib.pyplot as plt
from bfn.bfn_const import CONST_log_range, CONST_log_min, CONST_summary_rescale, CONST_exp_range, CONST_min_std_dev


def right_pad_dims_to(input_tensor, target):
    # pad input_tensor with 1s to match the dimensions of target
    padding_dims = target.ndim - input_tensor.ndim
    if padding_dims <= 0:
        return input_tensor
    return input_tensor.view(*input_tensor.shape, *((1,) * padding_dims))

def safe_log(data: Tensor):
    return data.clamp(min=CONST_log_min).log()

def safe_exp(data: Tensor):
    return data.clamp(min=-CONST_exp_range, max=CONST_exp_range).exp()

def sandwich(x: Tensor):
    return x.reshape(x.size(0), -1, x.size(-1))