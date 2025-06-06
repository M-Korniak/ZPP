import os
import torch
import pytest
from unittest import mock
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from src.trainer.trainer import (
    save_model, load_model, Trainer, AutoEncoderTrainer, setup_clearml
)
from src.model.model import SpatioTemporalTransformer, AutoEncoder, ModelArgs


@pytest.fixture
def dummy_modelargs():
    return ModelArgs()


@pytest.fixture
def dummy_model(dummy_modelargs):
    return SpatioTemporalTransformer(dummy_modelargs)


@pytest.fixture
def dummy_autoencoder(dummy_modelargs):
    return AutoEncoder(dummy_modelargs)


@pytest.fixture
def dummy_data():
    x = torch.randn(5, 16, 128)
    return DataLoader(x, batch_size=1)


@pytest.fixture
def dummy_data_autoencoder():
    x = torch.randn(5, 3, 64, 64)
    return DataLoader(x, batch_size=1)


def test_save_and_load_model(tmp_path, dummy_autoencoder, dummy_modelargs):
    dummy_autoencoder.encoder.fc = torch.nn.Linear(128, 64)
    dummy_autoencoder.decoder.fc = torch.nn.Linear(64, 128)
    dummy_autoencoder.set_decoder_init(False)

    save_path = tmp_path / "test_model.pth"
    save_model(dummy_autoencoder, dummy_modelargs, str(save_path))
    assert save_path.exists()

    loaded_model = load_model(str(save_path), 'AutoEncoder', device='cpu')
    assert isinstance(loaded_model, torch.nn.Module)



def test_compute_loss():
    trainer = Trainer()
    pred = torch.randn(4, 10)
    target = pred + 0.1
    loss = trainer.compute_loss(pred, target)
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor)


def test_get_optimizer_and_scheduler(dummy_model):
    trainer = Trainer()
    optimizer, scheduler = trainer.get_optimizer_and_scheduler(dummy_model.parameters())
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(scheduler, StepLR)  


def test_trainer_train_epoch(dummy_model, dummy_data):
    trainer = Trainer(n_epochs=1, batch_size=1)
    dummy_model.train = mock.Mock()
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)

    dummy_model.forward = mock.Mock(return_value=torch.randn(1, 15, 128))
    trainer.compute_loss = mock.Mock(return_value=torch.tensor(0.5, requires_grad=True))

    trainer.train_epoch(dummy_model, dummy_data, optimizer, epoch=1)


def test_trainer_evaluate(dummy_model, dummy_data):
    trainer = Trainer(batch_size=1)
    dummy_model.eval = mock.Mock()
    dummy_model.forward = mock.Mock(return_value=torch.randn(1, 15, 128))
    trainer.compute_loss = mock.Mock(return_value=torch.tensor(0.5))

    trainer.evaluate(dummy_model, dummy_data, epoch=1)


def test_autoencoder_train_epoch(dummy_autoencoder, dummy_data_autoencoder):
    trainer = AutoEncoderTrainer(n_epochs=1, batch_size=1)
    dummy_autoencoder.train = mock.Mock()
    optimizer = torch.optim.Adam(dummy_autoencoder.parameters(), lr=1e-3)

    dummy_autoencoder.forward = mock.Mock(return_value=torch.randn(1, 3, 64, 64))
    trainer.compute_loss = mock.Mock(return_value=torch.tensor(0.3, requires_grad=True))

    trainer.train_epoch(dummy_autoencoder, dummy_data_autoencoder, optimizer, epoch=1)


def test_autoencoder_evaluate(dummy_autoencoder, dummy_data_autoencoder):
    trainer = AutoEncoderTrainer()
    dummy_autoencoder.eval = mock.Mock()
    dummy_autoencoder.forward = mock.Mock(return_value=torch.randn(1, 3, 64, 64))
    trainer.compute_loss = mock.Mock(return_value=torch.tensor(0.25))

    trainer.evaluate(dummy_autoencoder, dummy_data_autoencoder, epoch=1)


def test_setup_clearml_env(monkeypatch):
    monkeypatch.setenv("CLEARML_ACCESS_KEY", "fake_key")
    monkeypatch.setenv("CLEARML_SECRET_KEY", "fake_secret")

    with mock.patch("clearml.Task.set_credentials") as mock_set:
        setup_clearml()
        mock_set.assert_called_once()
