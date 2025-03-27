import torch
import numpy as np
from torchvision import transforms
import imageio
import pytest
from unittest.mock import patch, MagicMock

from src.transformations.transformations import (
    load_gif,
    crop_to_field_of_view,
    normalize_image,
    unnormalize_image,
    resize_image,
    transform_image_to_trainable_form,
    custom_random_rotation,
    transformations_for_training,
    transformations_for_evaluation,
    transform_gif_to_tensor,
    MEANS,
    STDS
)

def test_load_gif_valid():
    fake_frames = np.random.randint(0, 256, (5, 64, 64, 3), dtype=np.uint8)
    
    with patch("imageio.get_reader") as mock_reader:
        mock_reader.return_value = fake_frames
        result = load_gif("fake_path.gif")
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 5, 3, 64, 64) 
    assert result.dtype == torch.float16

def test_load_gif_invalid_path():
    with patch("imageio.get_reader", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_gif("non_existent.gif")

def test_load_gif_empty_gif():
    with patch("imageio.get_reader") as mock_reader:
        mock_reader.return_value = []
        with pytest.raises(ValueError):
            load_gif("empty.gif")


def test_load_gif_incorrect_format():
    with patch("imageio.get_reader") as mock_reader:
        mock_reader.return_value = np.array([[[255, 255, 255]]])  
        with pytest.raises(ValueError):
            load_gif("incorrect.gif")

def test_crop_to_field_of_view_valid():
    image = torch.randn(1, 3, 600, 800)
    cropped_image = crop_to_field_of_view(image)
    
    assert cropped_image.shape == (1, 3, 461, 495)

def test_crop_to_field_of_view_edge_case():
    image = torch.randn(1, 3, 461 + 73, 495 + 101)
    cropped_image = crop_to_field_of_view(image)
    assert cropped_image.shape == (1, 3, 461, 495)

def test_crop_to_field_of_view_too_small():
    image = torch.randn(1, 3, 400, 400)
    
    with pytest.raises(IndexError):
        crop_to_field_of_view(image)

def test_normalize_image_valid():
    image = torch.randn(1, 3, 64, 64)
    mean = torch.tensor(MEANS).view(3, 1, 1)
    std = torch.tensor(STDS).view(3, 1, 1)
    normalized = normalize_image(image)
    
    assert normalized.shape == image.shape
    assert torch.allclose(normalized * std + mean, image, atol=1e-4)

def test_normalize_image_different_device():
    if torch.cuda.is_available():
        image = torch.randn(1, 3, 64, 64, device="cuda")
        normalized = normalize_image(image)
        assert normalized.device == image.device

def test_normalize_image_invalid_shape():
    image = torch.randn(1, 1, 64, 64) 
    with pytest.raises(ValueError):
        normalize_image(image)

def test_unnormalize_image_valid():
    image = torch.randn(1, 3, 64, 64)
    mean = torch.tensor(MEANS).view(3, 1, 1)
    std = torch.tensor(STDS).view(3, 1, 1)
    
    unnormalized = unnormalize_image((image - mean) / std)
    
    assert unnormalized.shape == image.shape
    assert torch.allclose(unnormalized, image, atol=1e-4)

def test_unnormalize_image_different_device():
    if torch.cuda.is_available():
        image = torch.randn(1, 3, 64, 64, device="cuda")
        unnormalized = unnormalize_image(image)
        assert unnormalized.device == image.device

def test_unnormalize_image_invalid_shape():
    image = torch.randn(1, 1, 64, 64) 
    with pytest.raises(ValueError):
        unnormalize_image(image)

def test_resize_image_valid():
    image = torch.randn(2, 3, 3, 128, 128)
    resized_image = resize_image(image, size=256)
    
    assert resized_image.shape == (2, 3, 3, 256, 256)

def test_resize_image_different_size():
    image = torch.randn(1, 5, 3, 64, 64)
    resized_image = resize_image(image, size=100)
    assert resized_image.shape == (1, 5, 3, 100, 100)

def test_resize_image_invalid_shape():
    image = torch.randn(1, 64, 64)  
    with pytest.raises(ValueError):
        resize_image(image)

def test_resize_image_different_device():
    if torch.cuda.is_available():
        image = torch.randn(1, 3, 3, 128, 128, device="cuda")
        resized_image = resize_image(image, size=256)
        assert resized_image.device == image.device

def test_transform_image_to_trainable_form_valid():
    image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    transformed = transform_image_to_trainable_form(image)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.dtype == torch.float32
    assert transformed.shape == image.shape

def test_transform_image_to_trainable_form_invalid_shape():
    image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)  
    with pytest.raises(ValueError):
        transform_image_to_trainable_form(image)

def test_transform_image_to_trainable_form_different_device():
    if torch.cuda.is_available():
        image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8, device="cuda")
        transformed = transform_image_to_trainable_form(image)
        assert transformed.device == image.device

def test_custom_random_rotation():
    image = torch.randn(3, 64, 64)  
    rotated_image = custom_random_rotation(image)
    
    assert isinstance(rotated_image, torch.Tensor)
    assert rotated_image.shape == image.shape  

def test_custom_random_rotation_values():
    image = torch.ones(3, 5, 5)  
    rotated_images = [(custom_random_rotation(image).numpy().flatten()) for _ in range(10)]
    assert len(rotated_images) > 1  

def test_custom_random_rotation_identity():
    image = torch.randn(3, 64, 64)
    rotated_image = custom_random_rotation(image)
    assert torch.allclose(image, rotated_image) or not torch.equal(image, rotated_image) 

def test_custom_random_rotation_device():
    if torch.cuda.is_available():
        image = torch.randn(3, 64, 64, device="cuda")
        rotated_image = custom_random_rotation(image)
        assert rotated_image.device == image.device

def test_transformations_for_training():
    image = torch.randn(1, 3, 256, 256)  
    transformed_image = transformations_for_training(image, crop_size=128)
    
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (1, 3, 128, 128)  

def test_transformations_for_training_different_crop():
    image = torch.randn(1, 3, 256, 256)
    transformed_image = transformations_for_training(image, crop_size=64)
    assert transformed_image.shape == (1, 3, 64, 64)

def test_transformations_for_training_randomness():
    image = torch.randn(1, 3, 256, 256)
    transformed_images = [(transformations_for_training(image, crop_size=128).numpy().flatten()) for _ in range(5)]
    assert len(transformed_images) > 1 

def test_transformations_for_evaluation():
    image = torch.randn(1, 3, 256, 256)
    transformed_image = transformations_for_evaluation(image, crop_size=128)
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (1, 3, 128, 128)

def test_transformations_for_evaluation_different_crop():
    image = torch.randn(1, 3, 256, 256)
    transformed_image = transformations_for_evaluation(image, crop_size=64)
    assert transformed_image.shape == (1, 3, 64, 64)

def test_transformations_for_evaluation_invalid_shape():
    image = torch.randn(1, 256, 256)
    with pytest.raises(ValueError):
        transformations_for_evaluation(image, crop_size=128)