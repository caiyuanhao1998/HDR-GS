import torch
from functools import lru_cache

from .modules.lpips import LPIPS


@lru_cache(maxsize=8)
def _get_criterion(net_type, version, device):
    return LPIPS(net_type, version).to(torch.device(device)).eval()


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = _get_criterion(net_type, version, str(device))
    return criterion(x, y)
