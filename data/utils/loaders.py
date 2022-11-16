import torch
from torch import Tensor
from torchvision.transforms import ToTensor

from typing import Tuple, Dict

import pathlib

from PIL import Image
import rawpy

from data.utils import xmp


def raw_loader(path: str) -> Tensor:
    """
    Returns tensor representation of image with shape (C, H, W)
    Args:
        path: Path to raw image file.

    Returns:
        tensor data representation of image.

    """
    rgb = rawpy.imread(path).postprocess()
    return torch.tensor(rgb).permute(2, 0, 1)


def image_loader(path: str) -> Image.Image:
    """Uses PIL to read an image and then returns that image as a tensor."""
    with Image.open(path) as im:
        temp = ToTensor()(im)
        return im


def xmp_loader(path: str) -> Tuple[Image.Image, Dict]:
    """
    Loads a PIL image from file along with a dictionary of meta-data for that file.

    The metadata must be in a directory "xmp" which at the level of the image's containing directory:

    data/
     |__images/
     |  |__ img1.jpg
     |
     |__xmp/
        |__ img1.xmp

    Args:
        path: Path to the image.

    Returns: The image and its meta-data.

    """
    xmp_path = pathlib.Path(path).parent.parent / "xmp"

    image = image_loader(path)
    meta_data = xmp.get_crs_tags(xmp_path.__str__())

    return image, meta_data
