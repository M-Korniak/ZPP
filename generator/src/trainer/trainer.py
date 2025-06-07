import os
from dataclasses import asdict
import torch
import torch.nn.functional as F
from typing import Iterable, Callable, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Logger, Task

import src.model.model as model
from src.model.model import ModelArgs 
import src.data_processing.data_processing as data_processing
import src.transformations.transformations as transformations
import src.visualizer.visualizer as visualizer
import src.generator.generator as generator

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


BATCH_NORM_TYPES = (
    torch.nn.BatchNorm1d
    | torch.nn.BatchNorm2d
    | torch.nn.BatchNorm3d
    | torch.nn.SyncBatchNorm
    | torch.nn.LazyBatchNorm1d
    | torch.nn.LazyBatchNorm2d
    | torch.nn.LazyBatchNorm3d
)


def setup_clearml():
    access_key = os.getenv('CLEARML_ACCESS_KEY')
    secret_key = os.getenv('CLEARML_SECRET_KEY')

    Task.set_credentials(
        web_host='https://app.clear.ml',
        api_host='https://api.clear.ml',
        files_host='https://files.clear.ml',
        key=access_key,
        secret=secret_key
    )


def save_model(trained_model: torch.nn.Module, args: ModelArgs, save_path: str) -> None:
    """
    Save the model's parameters and hyperparameters to a file.

    Args:
        trained_model (torch.nn.Module): The trained model.
        args (model.ModelArgs): The hyperparameters.
        save_path (str): Path to save the model parameters and hyperparameters.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'params': trained_model.state_dict(),
        'hyperparams': asdict(args)
    }, save_path)


def load_model(load_path: str, model_type: str, device: torch.device) -> torch.nn.Module:
    """
    Load model parameters from a file into a new model instance.

    Args:
        load_path (str): Path to the saved model parameters.
        model_type: either 'AutoEncoder' or 'SpatioTemporalTransformer'

    Returns:
        torch.nn.Module: The model with loaded parameters.
    """
    checkpoint = torch.load(load_path, map_location=device)
    params = checkpoint['params']
    hyperparams_dict = checkpoint['hyperparams']
    hyperparams = ModelArgs(**hyperparams_dict)

    if model_type == 'AutoEncoder':
        trained_model = model.AutoEncoder(hyperparams).to(device)

    elif model_type == 'SpatioTemporalTransformer':
        trained_model = model.SpatioTemporalTransformer(hyperparams).to(device)

    else:
        print("Incorrect model type; try again with one of the following : 'AutoEncoder', 'SpatioTemporalTransformer'.")
        return None

    encoder_in_features, encoder_out_features = params['encoder.fc.weight'].shape
    decoder_in_features, decoder_out_features = params['decoder.fc.weight'].shape

    trained_model.encoder.fc = torch.nn.Linear(encoder_out_features, encoder_in_features)
    trained_model.decoder.fc = torch.nn.Linear(decoder_out_features, decoder_in_features)
    trained_model.set_decoder_init(False)
    trained_model.load_state_dict(params)
    return trained_model


class Trainer:
    """
    Trainer class to train the model using AdamW optimizer and StepLR scheduler

    Args:
    - lr (float): The learning rate. Default 2e-4
    - weight_decay (float): The weight decay. Default 3e-5
    - batch_norm_momentum (float | None): The batch norm momentum. Default 0.01 - Important for training stability
    - n_epochs (int): The number of epochs. Default 10
    - device (str): The device to use. Default DEVICE
    - extra_augmentation (v2.Transform | None): The extra augmentation to use. Default None

    Attributes:
    - lr (float): The learning rate
    - weight_decay (float): The weight decay
    - batch_norm_momentum (float | None): The batch norm momentum
    - n_epochs (int): The number of epochs
    - device (str): The device to use
    - load_to_ram (bool): Whether to load the data to RAM
    - extra_augmentation (Optional[Callable]): The extra augmentation to use
    """
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_size: int = 16, batch_norm_momentum: float | None = 0.01, n_epochs: int = 10,
                 device: str = DEVICE, load_to_ram: bool = False,
                 extra_augmentation: Optional[Callable] = transformations.transformations_for_training,
                 data_folder: str = "../../data/tensors_to_load"):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.batch_norm_momentum = batch_norm_momentum
        self.n_epochs = n_epochs
        self.device = device
        self.load_to_ram = load_to_ram
        self.extra_augmentation = extra_augmentation
        self.data_folder = data_folder

    def get_optimizer_and_scheduler(
            self, parameters: Iterable[torch.nn.Parameter]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, fused=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
        return optimizer, lr_scheduler

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)

    def evaluate(self, model: torch.nn.Module, test_loader: DataLoader, epoch: int, logger: Logger = None):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(test_loader, desc="Evaluate")
        for batch in progress_bar:
            batch = batch.to(self.device)
            predictions = model(batch[:, :-1])
            loss = self.compute_loss(predictions, batch[:, 1:])

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        
        if logger is not None:
            logger.report_scalar(
                title="Validation Loss", series="Inner Transformer Loss", iteration=epoch, value=avg_loss
            )

    def train(self, model: torch.nn.Module, logger: Logger = None):
        model = model.to(self.device)

        train_loader, test_loader = (data_processing.
                                     get_dataloader(data_folder=self.data_folder,
                                                    batch_size=self.batch_size,
                                                    load_to_ram=self.load_to_ram,
                                                    transform=self.extra_augmentation))

        if self.batch_norm_momentum is not None:
            for m in model.modules():
                if isinstance(m, BATCH_NORM_TYPES):
                    m.momentum = self.batch_norm_momentum

        optimizer, lr_scheduler = self.get_optimizer_and_scheduler(model.parameters())

        for epoch in range(1, self.n_epochs + 1):
            self.train_epoch(model, train_loader, optimizer, epoch, logger)
            lr_scheduler.step()
            self.evaluate(model, test_loader, epoch, logger)

    def train_epoch(self, model: torch.nn.Module, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, epoch: int, logger: Logger = None):
        model.train()
        total_loss = 0.0
        batch_count = 0
        avg_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train epoch {epoch:>3}")
        for batch in progress_bar:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            predictions = model(batch[:, :-1])
            loss = self.compute_loss(predictions, batch[:, 1:])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
        if logger is not None:
            logger.report_scalar(
                title="Average Epoch Loss", series="Inner Transformer Loss",
                iteration=epoch, value=avg_loss
            )


class AutoEncoderTrainer(Trainer):
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_size: int = 16, batch_norm_momentum: float | None = 0.01, n_epochs: int = 10,
                 device: str = DEVICE,
                 extra_augmentation: Optional[Callable] = transformations.transformations_for_training):
        super().__init__(lr, weight_decay, batch_size, batch_norm_momentum, n_epochs, device, extra_augmentation, load_to_ram)

    def evaluate(self, model: torch.nn.Module, test_loader: DataLoader, epoch: int, logger: Logger = None):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(test_loader, desc="Evaluate")
        for batch in progress_bar:
            batch = batch.to(self.device)
            predictions = model(batch)
            loss = self.compute_loss(predictions, batch)

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if logger is not None:
            logger.report_scalar(
                title="Validation Loss", series="Inner Transformer Loss", iteration=epoch, value=avg_loss
            )

    def train_epoch(self, model: torch.nn.Module, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, epoch: int, logger: Logger = None):
        model.train()
        total_loss = 0.0
        batch_count = 0
        avg_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train epoch {epoch:>3}")
        for batch in progress_bar:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            predictions = model(batch)
            loss = self.compute_loss(predictions, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if logger is not None:
            logger.report_scalar(
                title="Average Epoch Loss", series="Inner Transformer Loss",
                iteration=epoch, value=avg_loss
            )


if __name__ == "__main__":
    trainer = Trainer(
        n_epochs=100,
        lr=2e-3,
        batch_size=4,
        load_to_ram=False,
        batch_norm_momentum=0.01,
        extra_augmentation=lambda image: transformations.transformations_for_training(image, crop_size=16)
    )
    args = model.ModelArgs()
    model = model.SpatioTemporalTransformer(args).to(DEVICE)
    trainer.train(model)

    train_loader, test_loader = data_processing.get_dataloader(
        batch_size=1,
        transform=lambda image: transformations.transformations_for_evaluation(image, crop_size=16)
    )
    model.eval().to(DEVICE)
    batch = next(iter(test_loader)).to(DEVICE)
    generated_video = generator.generate_time_lapse_from_tensor(model, batch[:, :100], video_length=258)
    generated_video = transformations.unnormalize_image(generated_video)
    visualizer.visualize_tensor_images_as_gif(generated_video[0], path="../../data/animation.gif")




