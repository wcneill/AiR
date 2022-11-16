import torch
from torch import nn


def crop(image: torch.Tensor, new_shape: int):
    if image.shape[-1] == new_shape:
        return image

    pad = image.shape[-1] - new_shape

    if pad % 2 == 0:
        pad = pad // 2
        return image[..., pad:-pad, pad:-pad]

    return image[..., :-pad, :-pad]