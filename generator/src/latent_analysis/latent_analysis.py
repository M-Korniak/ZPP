import os
import pandas as pd
import torch

import src.model.model as model

def load_metadata(csv_path):
    """
    Loads a CSV file containing metadata into a structured format (nested dictionary) indexed by Exp_ID and Image_Metadata_Site.
    The last dimension is Cell_Type.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        dict: A nested dictionary where t[exp_id][site] = cell_type.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Drop the Exp_Name column as it's not needed
    df = df.drop(columns=["Exp_Name"])

    # Create a nested dictionary structure
    metadata = {}
    for _, row in df.iterrows():
        exp_id = row["Exp_ID"]
        site = row["Image_Metadata_Site"]
        cell_type = row["Cell_Type"]

        if exp_id not in metadata:
            metadata[exp_id] = {}
        metadata[exp_id][site] = cell_type

    return metadata


def load_pt_files(folder_path):
    """
    Loads all .pt files in a folder into a structured tensor indexed by experiment ID and field of view.

    Args:
        folder_path (str): Path to the folder containing .pt files.

    Returns:
        dict: A nested dictionary where t[experiment_id][field_of_view] = tensor.
    """
    # Initialize the structure to store tensors
    tensors_dict = {}

    # Iterate over all .pt files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pt"):
            # Extract experiment ID and field of view from the filename
            parts = filename.split("_")
            experiment_id = int(parts[-3])  # Extracts the experiment ID
            field_of_view = int(parts[-1].split(".")[0])  # Extracts the field of view

            # Load the tensor from the .pt file
            file_path = os.path.join(folder_path, filename)
            tensor = torch.load(file_path)

            # Store the tensor in the dictionary
            if experiment_id not in tensors_dict:
                tensors_dict[experiment_id] = {}
            tensors_dict[experiment_id][field_of_view] = tensor

    return tensors_dict


# Latent space analysis
if __name__ == "__main__":
    metadata_path = "../../data/plateMap.csv"
    input_data_path = "../../data/tensors_to_load"
    
    extracted_metadata = load_metadata(metadata_path)
    print("Extracting metadata finished.")
    print(extracted_metadata)

    input_tensors = load_pt_files(input_data_path)
    print("Getting the input tensors finished.")
    print(input_tensors)