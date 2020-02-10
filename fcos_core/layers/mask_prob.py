# import torch
# from torch import nn
# from torch.autograd import Function

from fcos_core import _C

mask_prob_cuda = _C.maskprob_forward