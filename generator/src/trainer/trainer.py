import os
import torch
import torch.nn.functional as F
from typing import Iterable, Callable, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Logger, Task

import src.model.model as model
import src.data_processing.data_processing as data_processing
import src.transformations.transformations as transformations
import src.visualizer.visualizer as visualizer

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
    - extra_augmentation (v2.Transform | None): The extra augmentation to use
    """
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_size: int = 16, batch_norm_momentum: float | None = 0.01, n_epochs: int = 10,
                 device: str = DEVICE,
                 extra_augmentation: Optional[Callable] = transformations.transformations_for_training):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.batch_norm_momentum = batch_norm_momentum
        self.n_epochs = n_epochs
        self.device = device
        self.extra_augmentation = extra_augmentation

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
                                     get_dataloader(batch_size=self.batch_size,
                                                    transform=self.extra_augmentation))

        if self.batch_norm_momentum is not None:
            # Default torch.nn.BatchNorm2D.momentum is 0.1, but it's often too high.
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


if __name__ == "__main__":
    trainer = Trainer(
        n_epochs=20,
        lr=1e-4,
        batch_size=4,
        batch_norm_momentum=0.1,
        extra_augmentation=lambda image: transformations.transformations_for_training(image, crop_size=32)
    )
    args = model.ModelArgs()
    model = model.SpatioTemporalTransformer(args).to(DEVICE)
    trainer.train(model)

    # get the first batch of the loader
    train_loader, test_loader = data_processing.get_dataloader(
        batch_size=1,
        transform=lambda image: transformations.transformations_for_evaluation(image, crop_size=32)
    )

    model.eval()
    batch = next(iter(test_loader)).to(DEVICE)
    predictions = model(batch[:, :-1])
    predictions_unnormalized = transformations.unnormalize_image(predictions)
    visualizer.visualize_tensor_image(predictions_unnormalized[0][1])






