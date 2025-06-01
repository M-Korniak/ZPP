from typing import Tuple, Optional, Callable
import numpy as np
import torch
import shutil
from torch.utils.data import Dataset, DataLoader, random_split
import os
import argparse
import src.utils.utils as utils
import src.visualizer.visualizer as visualizer
import src.transformations.transformations as transformations

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


@torch.no_grad()
def load_experiment_data_to_tensor(
    experiments: Tuple[int] = (1, 2, 3, 4, 5, 6),
    maintain_experiment_visualization: bool = False,
    data_path: str = "../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz",
    experiments_path: str = "../../data/experiments",
    tensor_path: str = "../../data/tensors_to_load",
    custom_gif_path: Optional[str] = None
):
    """
    Transforms tabular data to images and loads them into a tensor of shape (B, S, C, H, W).
    Tensor is not normalized and is saved as float16 for memory efficiency.
    Clips 'ERKKTR_ratio' to [0.4, 2.7] to avoid outliers.
    Saves each field of view as a separate tensor file.

    Args:
    - experiments (Tuple[int]): Experiments to include in tensor. Default (1, 2, 3, 4, 5, 6).
    - maintain_experiment_visualization (bool): If True, keeps the visualizations of the experiments. Default False.
    - data_path (str): Path to the input CSV file. Default "../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz".
    - experiments_path (str): Path to save temporary experiment visualizations. Default "../../data/experiments".
    - tensor_path (str): Path to save the resulting tensors. Default "../../data/tensors_to_load".
    - custom_gif_path (Optional[str]): Custom path for saving GIFs. If None, uses default path. Default None.
    """
    os.makedirs(experiments_path, exist_ok=True)
    os.makedirs(tensor_path, exist_ok=True)
    if custom_gif_path:
        os.makedirs(custom_gif_path, exist_ok=True)

    df = utils.unpack_and_read(data_path)
    df['ERKKTR_ratio'] = np.clip(df['ERKKTR_ratio'], 0.4, 2.7)
    df = df[df['Exp_ID'].isin(experiments)]

    for experiment in experiments:
        df_experiment = df[df['Exp_ID'] == experiment]
        fields_of_view = np.sort(df_experiment['Image_Metadata_Site'].unique())
        experiments_tensor = torch.zeros(1, 258, 3, 256, 256, device=DEVICE, dtype=torch.float16)

        for field_of_view in fields_of_view:
            df_fov = df_experiment[df_experiment['Image_Metadata_Site'] == field_of_view]
            frames_count = df_fov['Image_Metadata_T'].max() + 1
            
            if custom_gif_path:
                gif_path = os.path.join(custom_gif_path, f"experiment_{experiment}_fov_{field_of_view}.gif")
            else:
                gif_path = os.path.join(experiments_path, f"experiment_{experiment}_fov_{field_of_view}.gif")
                
            visualizer.visualize_simulation(df_fov, number_of_frames=frames_count, path=gif_path)

            fov_tensor = transformations.transform_gif_to_tensor(gif_path).squeeze(0)

            if fov_tensor.shape[0] < 258:
                padding = torch.zeros(258 - fov_tensor.shape[0], 3, 256, 256, device=DEVICE)
                fov_tensor = torch.cat((fov_tensor, padding), dim=0)

            experiments_tensor[0] = fov_tensor

            if not maintain_experiment_visualization:
                os.remove(gif_path)

            tensor_save_path = os.path.join(tensor_path, f"experiments_tensor_exp_{experiment}_fov_{field_of_view}.pt")
            torch.save(experiments_tensor, tensor_save_path)

    if not maintain_experiment_visualization and os.path.exists(experiments_path):
        try:
            shutil.rmtree(experiments_path)
        except Exception as e:
            print(f"Nie udało się usunąć folderu {experiments_path}: {e}")


class TensorDataset(Dataset):
    def __init__(self, data_folder: str = "../../data/tensors_to_load/",
                 load_to_ram: bool = False,
                 transform: Optional[Callable] = None):
        """
        Args:
            data_folder (str): Path to the folder containing tensor files.
            load_to_ram (bool): If True, loads all tensors into RAM. Otherwise, loads lazily from disk.
        """
        self.data_folder = data_folder
        self.file_names = sorted(os.listdir(data_folder))
        self.file_names = [file for file in self.file_names if 'experiments_tensor' in file]
        self.load_to_ram = load_to_ram
        self.data_len = len(self.file_names)
        self.transform = transform

        if self.load_to_ram:
            self.data = []
            for f_name in self.file_names:
                file_path = os.path.join(data_folder, f_name)
                batches = torch.load(file_path)
                self.data.extend(batches)

            self.data = torch.stack(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.load_to_ram:
            item = self.data[idx]
        else:
            file_idx = self.file_names[idx]
            file_path = os.path.join(self.data_folder, file_idx)
            batches = torch.load(file_path)
            item = batches[0]

        if self.transform:
            item = self.transform(item)

        return item


def get_dataloader(data_folder: str = "../../data/tensors_to_load/",
                   load_to_ram: bool = False,
                   batch_size: int = 16,
                   num_workers: int = 0,
                   pin_memory: bool = False,
                   train_split: float = 0.8,
                   seed: int = 42,
                   transform: Optional[Callable] = None):
    """
    Get train and test DataLoaders for the TensorDataset.

    Args:
    - data_folder (str): Path to the folder containing tensor files.
    - load_to_ram (bool): If True, loads all tensors into RAM. Otherwise, loads lazily from disk.
    - batch_size (int): The number of samples in each batch.
    - num_workers (int): The number of workers to use for loading data.
    - pin_memory (bool): If True, copies Tensors into CUDA pinned memory before returning.
    - train_split (float): The fraction of data to be used for training (default: 80%).
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_dataloader (DataLoader): DataLoader for training set.
    - test_dataloader (DataLoader): DataLoader for testing set.
    """
    dataset = TensorDataset(data_folder, load_to_ram, transform=transform)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, test_dataloader

def main():
    parser = argparse.ArgumentParser(description='Process and visualize tensor data.')
    parser.add_argument('--tensor', type=str, help='Path to tensor file to visualize')
    parser.add_argument('--load-data', action='store_true', help='Load and process experiment data')
    parser.add_argument('--experiments', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='List of experiments to process')

    parser.add_argument('--data-path', type=str,
                        default='../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz',
                        help='Path to input CSV file')
    parser.add_argument('--tensor-path', type=str,
                        default='../../data/tensors_to_load',
                        help='Path to save or load tensor files')
    parser.add_argument('--custom-gif-path', type=str, default=None,
                        help='Custom path for saving GIFs (optional)')
    parser.add_argument('--visualize', action='store_true',
                        help='Flag to visualize first image from loaded tensor')
    
    args = parser.parse_args()
    
    if args.load_data:
        print("Loading and processing experiment data...")
        load_experiment_data_to_tensor(
            experiments=tuple(args.experiments),
            data_path=args.data_path,
            tensor_path=args.tensor_path,
            custom_gif_path=args.custom_gif_path
        )
        print("Data processing completed.")

    if args.tensor:
        print(f"Loading tensor from {args.tensor}")
        my_tensor = torch.load(args.tensor)

        if args.visualize:
            print("Visualizing first image in tensor...")
            visualizer.visualize_tensor_image(my_tensor[0][0])

if __name__ == "__main__":
    main()