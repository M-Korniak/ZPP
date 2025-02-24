import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import transforms

MEANS = [223.5459, 240.8361, 237.1875]
STDS = [64.2124, 38.7054, 44.2737]


def load_gif(path: str) -> torch.Tensor:
    """
    Loads a .gif file and returns the frames as a tensor.

    Args:
    - path (str): The path to the .gif file.

    Returns:
    - torch.Tensor: The tensor containing the frames of the .gif file. Shape: (B, S, C, H, W)
    """
    gif = imageio.get_reader(path)
    frames = np.array([frame for frame in gif])
    frames = np.transpose(frames, (0, 3, 1, 2))
    tensor_frames = torch.tensor(frames, dtype=torch.float16)  # Shape: (S, C, H, W)
    batched_tensor = tensor_frames.unsqueeze(0)  # Add batch dimension (B=1)
    return batched_tensor


def crop_to_field_of_view(image: torch.Tensor, upper_left: int = 73,
                          lower_right: int = 73 + 461, upper_right: int = 101,
                          lower_left: int = 101 + 495) -> torch.Tensor:
    """
    Crops the batched images tensor to the field of view.

    Args:
    - image (torch.Tensor): The image tensor to crop.

    Returns:
    - torch.Tensor: The cropped image tensor.
    """
    return image[..., upper_left:lower_right, upper_right:lower_left]


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalizes batched image tensor using the precomputed mean and std values.

    Args:
    - image (torch.Tensor): The image tensor to normalize.

    Returns:
    - torch.Tensor: The normalized image tensor.
    """
    mean = torch.tensor(MEANS, device=image.device).view(3, 1, 1)
    std = torch.tensor(STDS, device=image.device).view(3, 1, 1)
    return (image - mean) / std


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Unnormalizes batched image tensor using the precomputed mean and std values.

    Args:
    - image (torch.Tensor): The image tensor to unnormalize.

    Returns:
    - torch.Tensor: The unnormalized image tensor.
    """
    mean = torch.tensor(MEANS, device=image.device).view(3, 1, 1)
    std = torch.tensor(STDS, device=image.device).view(3, 1, 1)
    return image * std + mean


def resize_image(image: torch.Tensor, size: int = 256) -> torch.Tensor:
    """
    Resizes batched image tensor to a square size.

    Args:
    - image (torch.Tensor): The image tensor to resize.
    - size (int): The size to resize the image to.

    Returns:
    - torch.Tensor: The resized image tensor.
    """
    B, S = image.shape[:2]
    image = image.view(B * S, *image.shape[2:])
    resize_transform = transforms.Resize((size, size), antialias=False)
    image = resize_transform(image)
    return image.view(B, S, *image.shape[1:])


@torch.no_grad()
def transform_gif_to_tensor(gif_path: str = "../../data/simulation.gif") -> torch.Tensor:
    """
    Transforms a .gif file to a normalized, cropped tensor.

    Args:
    - gif_path (str): The path to the .gif file.

    Returns:
    - torch.Tensor: The tensor containing the frames of the .gif file. Shape: (B, S, C, H, W)
    """
    frames = load_gif(gif_path)
    frames = crop_to_field_of_view(frames)
    frames = resize_image(frames)
    return frames











