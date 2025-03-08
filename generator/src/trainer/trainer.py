import os
import torch
import torch.nn.functional as F
from typing import Iterable, Callable, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Logger, Task
import matplotlib.pyplot as plt
import src.model.model as model
import src.data_processing.data_processing as data_processing
import src.transformations.transformations as transformations

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class Trainer:
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_size: int = 16, n_epochs: int = 10, device: str = DEVICE,
                 extra_augmentation: Optional[Callable] = transformations.transformations_for_training):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.extra_augmentation = extra_augmentation

    def get_optimizer_and_scheduler(self, parameters: Iterable[torch.nn.Parameter]):
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, fused=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
        return optimizer, scheduler

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        return F.mse_loss(predictions, targets)

    def save_images(self, original: torch.Tensor, reconstructed: torch.Tensor, config: dict):
        # Pobieramy tylko pierwszy obrazek z batcha
        original = original[0]  # (3, H, W)
        reconstructed = reconstructed[0]  # (3, H, W)

        # Zapis obrazu przed odnormalizowaniem (oryginalny i rekonstruowany)
        original_before = original.cpu().detach()
        reconstructed_before = reconstructed.cpu().detach()

        # Odwrócenie normalizacji
        original = transformations.unnormalize_image(original.cpu().detach())
        reconstructed = transformations.unnormalize_image(reconstructed.cpu().detach())

        # Zamiana wymiarów na (H, W, C) do wyświetlenia
        original_before = original_before.permute(1, 2, 0).numpy()  # (H, W, 3)
        reconstructed_before = reconstructed_before.permute(1, 2, 0).numpy()  # (H, W, 3)
        original = original.permute(1, 2, 0).numpy()  # (H, W, 3)
        reconstructed = reconstructed.permute(1, 2, 0).numpy()  # (H, W, 3)

        # Tworzymy wykres z obrazem przed i po odnormalizowaniu
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # Przed odnormalizowaniem
        axes[0, 0].imshow(original_before)
        axes[0, 0].set_title("Original Before Normalization")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(reconstructed_before)
        axes[0, 1].set_title("Reconstruction Before Normalization")
        axes[0, 1].axis("off")

        # Po odnormalizowaniu
        axes[1, 0].imshow(original)
        axes[1, 0].set_title("Original After Normalization")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(reconstructed)
        axes[1, 1].set_title("Reconstruction After Normalization")
        axes[1, 1].axis("off")

        # Generowanie nazwy pliku
        config_str = f"dim_{config['dim']}_outch_{'_'.join(map(str, config['out_channels']))}"
        save_path = f"reconstruction_{config_str}.png"
        
        plt.savefig(save_path)
        plt.close()
        print(f"Saved reconstruction: {save_path}")



    def evaluate(self, model: torch.nn.Module, test_loader: DataLoader, epoch: int, config: dict, logger: Logger = None):
        model.eval()
        total_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(test_loader, desc="Evaluate")
        with torch.no_grad():
            for batch in progress_bar:
                batch = batch.to(self.device)
                print(batch.shape)

                for i in range(batch.shape[1]):  # batch.shape[1] = 258

                    sub_batch = batch[0, i:i+1].unsqueeze(0)  # Wybieramy pojedynczy obraz w kształcie (1, 1, 3, 128, 128)
                    
                    reconstructions = model(sub_batch)
                    loss = self.compute_loss(reconstructions, sub_batch)

                    total_loss += loss.item()
                    batch_count += 1
                    avg_loss = total_loss / batch_count
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

            batch = next(iter(test_loader)).to(self.device)
            reconstructions = model(batch[0, i:i+1].unsqueeze(0))
            # print("zapisałem obrazki")
            self.save_images(batch[0], reconstructions[0], config)

        if logger is not None:
            logger.report_scalar("Validation Loss", "Autoencoder Loss", epoch, avg_loss)

    def train(self, model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: dict, logger: Logger = None):
        model = model.to(self.device)
        optimizer, scheduler = self.get_optimizer_and_scheduler(model.parameters())

        for epoch in range(1, self.n_epochs + 1):
            model.train()
            total_loss = 0.0
            batch_count = 0
            progress_bar = tqdm(train_loader, desc=f"Train epoch {epoch:>3}")

            for batch in progress_bar:
                batch = batch.to(self.device)
                # print(batch.shape)
                for i in range(batch.shape[1]):  # batch.shape[1] = 258
                    sub_batch = batch[0, i:i+1].unsqueeze(0) # Wybieramy pojedynczy obraz w kształcie (1, 1, 3, 128, 128)
                    # print("sub_batch shape !!!")
                    # print(sub_batch.shape)
                    optimizer.zero_grad()
                    reconstructions = model(sub_batch)
                    loss = self.compute_loss(reconstructions, sub_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1
                    avg_loss = total_loss / batch_count
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

            scheduler.step()
            if epoch == self.n_epochs:
                self.evaluate(model, test_loader, epoch, config, logger)

            if logger is not None:
                logger.report_scalar("Training Loss", "Autoencoder Loss", epoch, avg_loss)


if __name__ == "__main__":
    # data_processing.load_experiment_data_to_tensor()

    trainer = Trainer(n_epochs=25, lr=1e-4, batch_size=1)

    outer_channels_list = [[32, 64, 128, 256], [64, 64, 128, 128], [32, 32, 64, 64], [32, 32, 32, 32], [32, 32, 64, 64]]
    dim_list = [32, 64, 128, 256]

    for out_channels in outer_channels_list:
        for dim in dim_list:
            # tworze model
            args = model.ModelArgs(dim=dim, out_channel_sizes=out_channels)
            autoencoder = model.AutoEncoder(args).to(DEVICE)

            train_loader, test_loader = data_processing.get_dataloader(
                batch_size=trainer.batch_size,
                transform=transformations.transformations_for_training
            )
            

            config = {"dim": dim, "out_channels": out_channels}
            # print(f"Training with config: {config}")
            trainer.train(autoencoder, train_loader, test_loader, config)
