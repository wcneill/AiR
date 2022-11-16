import math
import pathlib
import os
from typing import Optional, Tuple, Callable, List, Any
from torch.utils.data import Dataset

from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.transforms import ToTensor
from torch import Tensor

import matplotlib.pyplot as plt
from PIL import Image

import data.utils.xmp as xmp


def show_tensor_images(image_tensor):
    """
    Function for visualizing images: Given a tensor of images, plots each image.

    Args:
        image_tensor: (B, n_channels, H, W) - A tensor containing B images with C channels each.
    """

    n_images = image_tensor.shape[0]
    n_rows = int(math.ceil(n_images / 4))
    n_cols = min(n_images, int(math.ceil(n_images / n_rows)))

    _, axes = plt.subplots(n_rows, n_cols)

    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]

    for im, ax in zip(image_tensor, axes):
        ax.imshow(im.detach().cpu().permute(1, 2, 0))

    plt.show()


def make_dataset(directory: str, extensions) -> List[str]:
    """
    Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory: Top level location of files in question.
        extensions: Allowed extensions for the dataset being created, i.e. jpg, gif, png etc.

    Returns:
        List of absolute paths to files.

    """

    directory = os.path.expanduser(directory)

    instances = []
    folder = pathlib.Path(directory)

    for f in sorted(folder.iterdir()):
        path = f.absolute().resolve()
        if f.is_file() and is_valid_file(path.__str__(), extensions):
            instances.append(path.__str__())

    return instances


def is_valid_file(path_string: str, extensions: Optional[Tuple[str]]) -> bool:
    if extensions is None:
        return True
    return has_file_allowed_extension(path_string, extensions)


def cr3_loader(path: str) -> Tensor:
    pass


def image_loader(path: str) -> Image:
    with Image.open(path) as im:
        temp = ToTensor()(im)
        return im

def xmp_loader(path: str) -> Tuple[Image, dict]:
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


class AutoImageData(Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            valid_extensions: Optional[Tuple[str, ...]] = None,
            input_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Dataset of images.

        Indexing into the dataset will return a tuple of `(original_img, xformed_img)` in the case that a target transform
        is requested. Otherwise, the second element of the tuple will be None.

        Args:
            root:
            loader:
            valid_extensions:
            input_transform:
            target_transform:
        """

        self.root = root
        self.loader = loader
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.samples = make_dataset(root, valid_extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.input_transform is not None:
            sample = self.input_transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(sample)
        else:
            target = None

        return sample, target

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "input_transform") and self.input_transform is not None:
            body += [repr(self.input_transform)]
        if hasattr(self, "target_transform") and self.target_transform is not None:
            body += [repr(self.input_transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""


class MetaImageSet(AutoImageData):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            valid_extensions: Optional[Tuple[str, ...]] = None,
            input_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """

        Args:
            root:
            loader: Loader is a callable that takes the argument of a path to an image. The loader is responsible for
                returning both the image and the meta-data based on the input path.
            valid_extensions:
                Image extensions allowed for this dataset.
            input_transform:
                Any transform to apply to the image.
            target_transform:
                Any transform to apply to the meta-data.
        """

        super().__init__(root, loader, valid_extensions, input_transform, target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, metadata)
        """
        path = self.samples[index]
        image, meta = self.loader(path)

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            meta = self.target_transform(meta)

        return image, meta
