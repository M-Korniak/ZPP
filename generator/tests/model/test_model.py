# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# This file has been modified from the original Llama 3 source code.

import os
import math
import numpy as np
from itertools import product
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Iterable
from clearml import Logger, Task
# uncomment if running in colab:
# from google.colab import userdata
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
from torch import nn

import src.model.model as model

# Constants definiton
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 128
SEQ_LEN = 64
VOCAB_SIZE = 7
NEXT_PROB = .1
INITIAL = 2

@dataclass
class ModelArgs:
    vocab_size: int = VOCAB_SIZE # Added!
    
    dim: int = 256  # Play to determine the best value
    n_layers: int = 128
    n_heads: int = 8
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    rope_theta: float = 21000

    max_batch_size: int = 32
    max_seq_len: int = 258

    out_channel_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    scaling_factor: int = 2


# Modified inner transformer model defintion
class InnerTransformer(nn.Module):
    """
    Transformer model with multiple model blocks.

    Args:
    - params (ModelArgs): The model arguments

    Attributes:
    - params (ModelArgs): The model arguments
    - n_layers (int): The number of layers
    - layers (torch.nn.ModuleList): The list of model blocks
    - norm (RMSNorm): The RMSNorm layer
    - freqs_cis (torch.Tensor): The precomputed frequencies
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        # Added : embedding
        self.embedding = torch.nn.Embedding(params.vocab_size, params.dim)
        # -----------------

        self.transformer = model.Transformer(params)

        # Added : final projection (into distribution over vocabulary)
        self.final_proj = torch.nn.Linear(params.dim, params.vocab_size)
        # -----------------

    def forward(self, input_tensor: torch.Tensor, start_pos: int = 0):
        # Added : embedding
        input_tensor = self.embedding(input_tensor) # (B, S) -> (B, S, dim)
        # -----------------
        
        h = self.transformer(input_tensor)
        
        # Added : final projection
        h = self.final_proj(h)
        # -----------------
        return h

# II. Adjusted trainer

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
    # Comment if running in colab:
    access_key = os.getenv('CLEARML_ACCESS_KEY')
    secret_key = os.getenv('CLEARML_SECRET_KEY')
    # Uncomment if running in colab:
    # access_key = userdata.get('CLEARML_ACCESS_KEY')
    # secret_key = userdata.get('CLEARML_SECRET_KEY')

    Task.set_credentials(
        web_host='https://app.clear.ml',
        api_host='https://api.clear.ml',
        files_host='https://files.clear.ml',
        key=access_key,
        secret=secret_key
    )


class TransformerTrainer:
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_norm_momentum: float | None = 0.002, n_epochs: int = 10,
                 device: str = DEVICE, extra_augmentation: v2.Transform | None = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_norm_momentum = batch_norm_momentum
        self.n_epochs = n_epochs
        self.device = device
        self.extra_augmentation = extra_augmentation  # TODO: Add reasonable extra augmentation

    def get_optimizer_and_scheduler(
            self, parameters: Iterable[torch.nn.Parameter]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, fused=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return optimizer, lr_scheduler

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)

    def evaluate(self, model: torch.nn.Module, test_loader: DataLoader, epoch: int, logger: Logger = None):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(enumerate(test_loader), desc="Evaluate")
        for i, (batch, y) in progress_bar:
            batch = batch.to(self.device)
            #batch = transformations.transform_image_to_trainable_form(batch)
            predictions = model(batch[:, :-1])

            #loss = self.compute_loss(predictions, batch[:, 1:])
            loss = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), batch[:, 1:].reshape(-1))

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if logger is not None:
            logger.report_scalar(
                title="Validation Loss", series="Inner Transformer", iteration=epoch, value=avg_loss
            )

    def evaluate_accuracy(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, epoch: int, logger: Logger = None):
        model.eval()
        sum_acc = 0
        num_examples = 0

        progress_bar = tqdm(enumerate(dataloader), desc="Evaluate")
        for i, (batch, y) in progress_bar:
            batch = batch.to(self.device)
            #batch = transformations.transform_image_to_trainable_form(batch)
            model_out = model(batch[:, :-1])
            y = batch[:, 1:]

            acc = (torch.argmax(model_out, dim=-1) == y).to(torch.float32).sum()
            sum_acc += acc
            num_examples += model_out.shape[0] * model_out.shape[1]

        avg_acc = sum_acc / num_examples
        # Added : logging the evaluated accuracy (after each training epoch)
        if logger is not None:
            logger.report_scalar(
                title="Validation Accuracy", series="Inner Transformer", iteration=epoch, value=avg_acc
            )

        return avg_acc

    def train(self, model: torch.nn.Module, train_loader: DataLoader,
              test_loader: DataLoader, logger: Logger = None):
        model = model.to(self.device)

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

            # Evaluating accuracy after each epoch
            acc = self.evaluate_accuracy(model, test_loader, epoch, logger)
            print(f"{epoch}: Avg eval accuracy {acc}")

    def train_epoch(self, model: torch.nn.Module, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, epoch: int, logger: Logger = None):
        model.train()
        total_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(enumerate(train_loader), desc=f"Train epoch {epoch:>3}")
        for i, (batch, y) in progress_bar:
            batch = batch.to(self.device)
            #batch = transformations.transform_image_to_trainable_form(batch)

            optimizer.zero_grad()
            predictions = model(batch[:, :-1])

            #loss = self.compute_loss(predictions, batch[:, 1:])
            loss = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if logger is not None:
            logger.report_scalar(
                title="Average Epoch Loss", series="Inner Transformer", iteration=epoch, value=avg_loss
            )


# III. Artificial data generation

def generate_recursive(n_first, vocab_size, next_prob):
    assert 0 < vocab_size
    initial = np.random.randint(0, vocab_size, n_first)
    coeffs = np.random.randint(0, vocab_size, n_first)

    return initial, coeffs, vocab_size, next_prob

class SeqGen:
    """
    For generating recurrent sequences with stochastically repeating terms.
    """
    def __init__(self, initial, coeffs, size, next_prob):
        assert len(coeffs) == len(initial)
        self.initial = initial
        self.coeffs = coeffs
        self.size = size
        self.next_prob = next_prob

        self.current = initial

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.random() < self.next_prob:
          new = self.current[-1] + 1
        else:
          new = (self.current @ self.coeffs)

        new %= self.size
        self.current = np.append(self.current, new)[1:]

        return new

    def __key(self):
        return (tuple(self.initial), tuple(self.coeffs), self.size, self.next_prob)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SeqGen):
            return self.__key() == other.__key()


def generate_dataset(gen_factory, seq_len, num_entries, exclude = []):
    """
    For generating datasets with num_entries elements each
    of length seq_len.

      gen_factory is a procedure that returns
        instance of SeqGen when called.

      seq_len is the length of the sequence to generate.

      num_entries is the number of sequences to generate.

      exclude is the set of sequences that aren't to be used in training
    """
    entries = []
    generators = []
    for e in range(num_entries):
        while True:
          seq_gen = gen_factory()
          if seq_gen in exclude:
              continue

          seq = []
          for s in range(seq_len + 1):
              seq.append(next(seq_gen))

          break

        generators.append(seq_gen)
        entries.append(seq)
    data = torch.tensor(entries, dtype=torch.long)
    # I put data as both x and y due to the way data are handled in Trainer
    return torch.utils.data.TensorDataset(data, data), set(generators)

def example_generator(gen):
    """
      A procedure that returns a representation of
      a single data entrance.
    """
    def example_gen():
        return SeqGen(*gen())
    return example_gen


PERM_EXAMPLE_GENERATOR = example_generator(lambda: generate_recursive(INITIAL, VOCAB_SIZE, NEXT_PROB))

TEST_DATASET, generators = generate_dataset(
    gen_factory=PERM_EXAMPLE_GENERATOR, seq_len=SEQ_LEN, num_entries=1000)
TRAIN_DATASET, _ = generate_dataset(
    gen_factory=PERM_EXAMPLE_GENERATOR, seq_len=SEQ_LEN, num_entries=10000, exclude=generators)


TRAIN_LOADER = torch.utils.data.DataLoader(
    TRAIN_DATASET, batch_size=BATCH_SIZE)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE)


# IV. Training & evaluation

def test_hyperparameters(dim, n_layers, n_heads, rope_theta, n_epochs, batch_size, lr, use_clearml):
    # Optional: If use_clearml is True, set up ClearML
    if use_clearml:
        setup_clearml()

    # Get the data loaders
    train_loader, test_loader = TRAIN_LOADER, TEST_LOADER

    # Set up the model args from ModelArgs dataclass
    args = ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, rope_theta=rope_theta)
    # Intialize the trainer and model
    model = InnerTransformer(args).to(DEVICE)
    trainer = TransformerTrainer(n_epochs=n_epochs, lr=lr)

    # Create a new ClearML task for each run
    params = {
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'rope_theta': rope_theta,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'lr': lr,
    }

    if use_clearml:
        task = Task.init(
            project_name='Test Transformer',
            task_name=f'Run {n_epochs} - ' + ', '.join([f'{key}: {value}' for key, value in params.items()]),
            task_type=Task.TaskTypes.optimizer
        )
        # Log hyperparameters for this task
        task.connect(params)
        logger = task.get_logger()
    else:
        logger = None

    # Train the model
    trainer.train(model, train_loader, test_loader, logger=logger)

    if use_clearml:
        # Close the experiment with this hiperparameters.
        task.close()

# Define hyperparameter ranges
dim_values = [32, 64, 128, 256]
n_layers_values = [4, 8, 16, 32, 64]
n_heads_values = [4, 8, 16, 32]
rope_theta_values = [100, 1000, 10000, 20000]
n_epochs_values = [20]
batch_size_values = [32, 64, 128]
lr_values = [1e-2, 1e-3, 2e-4]
use_clearml_values = [False] # Change to True to activate clearml logging!

# Generate all combinations of hyperparameters
combinations = product(
    dim_values, n_layers_values, n_heads_values, rope_theta_values,
    n_epochs_values, batch_size_values, lr_values, use_clearml_values
)

# Iterate through all combinations and run the test
for i, (dim, n_layers, n_heads, rope_theta, n_epochs, batch_size, lr, use_clearml) in enumerate(combinations):
    print(f"Running combination {i + 1}:")
    print(f"  dim: {dim}, n_layers: {n_layers}, n_heads: {n_heads}, rope_theta: {rope_theta}")
    print(f"  n_epochs: {n_epochs}, batch_size: {batch_size}, lr: {lr}, use_clearml: {use_clearml}")
    print("-" * 50)

    # Call the test function
    test_hyperparameters(dim, n_layers, n_heads, rope_theta, n_epochs, batch_size, lr, use_clearml)

    print("Test completed.\n")