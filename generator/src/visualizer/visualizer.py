import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
from tqdm import tqdm
import os


def visualize_tensor_image(image_tensor: torch.tensor):
    """
    Visualizes a tensor image shape (C, H, W) as a matplotlib plot without any axis.
    Tensors are floats from range [0, 255.0].

    Args:
    - image_tensor (torch.tensor): The tensor image to visualize.

    Returns:
    - None
    """
    image = image_tensor.permute(1, 2, 0).detach().float().cpu().numpy() / 255.0
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def visualize_tensor_images_as_gif(image_tensors: torch.tensor, path: str = "../../data/animation.gif") -> None:
    """
    Visualizes a list of tensor images as a gif.

    Args:
    - image_tensors (torch.tensor): The list of tensor images to visualize.
    - path (str): The path to save the gif.

    Returns:
    - None
    """
    images = (image_tensor.permute(1, 2, 0).clip(0., 255.0)
              .detach().float().cpu().numpy() / 255.0 for image_tensor in image_tensors)

    with imageio.get_writer(path, mode='I', duration=0.1) as writer:
        for image in images:
            writer.append_data((image * 255).astype(np.uint8))


def visualize_simulation(simulation: pd.DataFrame, number_of_frames: int = 258,
                         path: str = "../../data/simulation.gif",
                         temp_frames_dir: str = "../../data/temp_frames") -> None:
    """
    Visualizes the simulation of nuclei movement and ERK values over time.

    Args:
    - simulation (pd.DataFrame): The DataFrame containing the simulation data.
    - number_of_frames (int): The number of frames to simulate.
    - path (str): Path to save the final GIF.
    - temp_frames_dir (str): Directory to store temporary frames.

    Returns:
    - None
    """
    marker = 'ERKKTR_ratio'

    min_value = np.log(0.4)
    max_value = np.log(2.7)

    os.makedirs(temp_frames_dir, exist_ok=True)
    frames = []

    colors = ['darkblue', 'blue', 'turquoise', 'yellow', 'orange', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('DarkBlueToYellow', colors)

    for t in tqdm(range(0, number_of_frames), desc='Creating frames'):
        current_frame = simulation[simulation['Image_Metadata_T'] == t]
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            current_frame['objNuclei_Location_Center_X'],
            current_frame['objNuclei_Location_Center_Y'],
            s=16,
            c=np.log(current_frame[marker]),
            cmap=custom_cmap,
            vmin=min_value,
            vmax=max_value
        )

        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.xlabel('Center X')
        plt.ylabel('Center Y')
        plt.title(f'Simulating Movement Time: {t}')
        plt.grid(False)

        cbar = plt.colorbar(sc, label=f'Intensity ({marker})')

        filename = os.path.join(temp_frames_dir, f"frame_{t:03d}.png")
        plt.savefig(filename, transparent=False)
        frames.append(filename)
        plt.close()

    with imageio.get_writer(path, mode='I', duration=1) as writer:
        for frame in frames:
            image = imageio.imread(frame)[..., :3]
            writer.append_data(image)

    for frame in frames:
        os.remove(frame)

    os.rmdir(temp_frames_dir)






