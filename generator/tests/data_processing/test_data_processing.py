import pytest
import torch
import tempfile
import os
from unittest import mock
from src.data_processing.data_processing import (
    load_experiment_data_to_tensor,
    TensorDataset,
    get_dataloader
)

@mock.patch("src.transformations.transformations.transform_gif_to_tensor", return_value=torch.rand(1, 258, 3, 256, 256))
@mock.patch("src.visualizer.visualizer.visualize_simulation")
def test_load_experiment_data_to_tensor(mock_visualize, mock_transform):
    with tempfile.TemporaryDirectory() as temp_data, tempfile.TemporaryDirectory() as tensor_out:
        dummy_csv = os.path.join(temp_data, "dummy.csv")

        # Dummy CSV with minimal data
        import pandas as pd
        df = pd.DataFrame({
            'Exp_ID': [1] * 5,
            'Image_Metadata_Site': [1] * 5,
            'Image_Metadata_T': [0, 1, 2, 3, 4],
            'track_id': [0, 1, 2, 3, 4],
            'objNuclei_Location_Center_X': [10, 20, 30, 40, 50],
            'objNuclei_Location_Center_Y': [60, 70, 80, 90, 100],
            'ERKKTR_ratio': [0.3, 0.5, 2.0, 3.0, 1.5]
        })
        df.to_csv(dummy_csv, index=False)

        # Patch utils.unpack_and_read to load our dummy CSV
        with mock.patch("src.utils.utils.unpack_and_read", return_value=pd.read_csv(dummy_csv)):
            load_experiment_data_to_tensor(
                experiments=(1,),
                data_path=dummy_csv,
                tensor_path=tensor_out,
                custom_gif_path=temp_data,
                maintain_experiment_visualization=False
            )

            files = os.listdir(tensor_out)
            assert any(f.endswith(".pt") for f in files), "No .pt tensor file was saved."

def test_tensor_dataset_loading(tmp_path):
    # Save dummy tensor file
    dummy_tensor = torch.rand(1, 258, 3, 256, 256)
    file_path = tmp_path / "experiments_tensor_exp_1_fov_1.pt"
    torch.save(dummy_tensor, file_path)

    dataset = TensorDataset(data_folder=str(tmp_path))
    assert len(dataset) == 1
    assert isinstance(dataset[0], torch.Tensor)
    assert dataset[0].shape == (258, 3, 256, 256)

def test_tensor_dataset_ram_loading(tmp_path):
    dummy_tensor = torch.rand(1, 258, 3, 256, 256)
    file_path = tmp_path / "experiments_tensor_exp_1_fov_1.pt"
    torch.save(dummy_tensor, file_path)

    dataset = TensorDataset(data_folder=str(tmp_path), load_to_ram=True)
    assert len(dataset) == 1
    assert isinstance(dataset[0], torch.Tensor)

def test_get_dataloader(tmp_path):
    dummy_tensor = torch.rand(1, 258, 3, 256, 256)
    file_path = tmp_path / "experiments_tensor_exp_1_fov_1.pt"
    torch.save(dummy_tensor, file_path)

    train_loader, test_loader = get_dataloader(
        data_folder=str(tmp_path),
        load_to_ram=True,
        batch_size=1,
        train_split=0.5
    )

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch, torch.Tensor)
    assert train_batch.shape == (1, 258, 3, 256, 256)
