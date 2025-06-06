import os
import shutil
import pytest
import torch
import pandas as pd
import numpy as np
from unittest import mock

from src.visualizer.visualizer import (
    visualize_tensor_image,
    visualize_tensor_images_as_gif,
    visualize_simulation
)


@pytest.fixture
def sample_tensor_image():
    return torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)


@pytest.fixture
def sample_tensor_images():
    return torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)


@pytest.fixture
def sample_simulation_dataframe():
    data = {
        'Image_Metadata_T': np.repeat(np.arange(3), 10),
        'objNuclei_Location_Center_X': np.random.uniform(0, 1024, 30),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 1024, 30),
        'ERKKTR_ratio': np.random.uniform(0.5, 2.5, 30),
    }
    return pd.DataFrame(data)


def test_visualize_tensor_image(sample_tensor_image):
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        visualize_tensor_image(sample_tensor_image)
        mock_show.assert_called_once()


def test_visualize_tensor_images_as_gif(tmp_path, sample_tensor_images):
    gif_path = tmp_path / "test_animation.gif"
    visualize_tensor_images_as_gif(sample_tensor_images, path=str(gif_path))
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_visualize_simulation(tmp_path, sample_simulation_dataframe):
    gif_path = tmp_path / "simulation.gif"
    frames_dir = tmp_path / "temp_frames"

    visualize_simulation(
        simulation=sample_simulation_dataframe,
        number_of_frames=3,
        path=str(gif_path),
        temp_frames_dir=str(frames_dir)
    )

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    assert not os.path.exists(frames_dir)
