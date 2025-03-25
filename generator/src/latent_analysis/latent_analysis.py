import os
import pandas as pd
import numpy as np
import torch
   
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.data_processing.data_processing import load_experiment_data_to_tensor
import src.trainer.trainer as trainer
from src.trainer.trainer import load_model
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
    path_to_model = "../../data/saved_model64_200_alternative.pth"
    #load_experiment_data_to_tensor()
    
    # Loading experiments metadata
    extracted_metadata = load_metadata(metadata_path)
    print("Extracting metadata finished.")
    print(extracted_metadata)

    input_tensors = load_pt_files(input_data_path)
    print("Getting the input tensors finished.")
    
    # Loading the trained model
    trained_model = load_model(path_to_model, 'SpatioTemporalTransformer', torch.device("cpu"))
    print("Loading the pre-trained model finished.")
    trained_model.eval()

    latent_vectors = []
    cell_types = []
    exp_ids_list = []

    # Getting the latent vectors
    for exp_id in input_tensors:
        for fov in input_tensors[exp_id]:
            # Check metadata existence
            if exp_id not in extracted_metadata or fov not in extracted_metadata[exp_id]:
                continue
            # Get metadata
            cell_type = extracted_metadata[exp_id][fov]
            
            # Process tensor
            tensor = input_tensors[exp_id][fov].unsqueeze(0)
            with torch.no_grad():
                latent = trained_model.get_encoder_latent_space(tensor).flatten().cpu().numpy()
            
            # Store results
            latent_vectors.append(latent)
            cell_types.append(cell_type)
            exp_ids_list.append(exp_id)
    # TODO: add the same visualisation for transformer space too

    latent_matrix = np.array(latent_vectors)

    # TODO: try different components number
    # Perform PCA
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(latent_matrix)

    df_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(5)])
    df_pca['Cell_Type'] = cell_types

    # Plot PC1 vs PC2
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cell_Type', data=df_pca, palette='viridis')
    plt.title('PCA of Latent Space - PC1 vs PC2')
    plt.show()

    # Additional visualisation using umap
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(latent_matrix)

    df_umap = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    df_umap['Cell_Type'] = cell_types

    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cell_Type', data=df_umap, palette='viridis')
    plt.title('UMAP Visualization of Latent Space')
    plt.show()
    
    # Classification Accuracy
    X_train, X_test, y_train, y_test = train_test_split(latent_matrix, cell_types, test_size=0.2)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(f"Classification Accuracy: {clf.score(X_test, y_test):.2f}")

    # Silhouette Score
    le = LabelEncoder()
    labels = le.fit_transform(cell_types)
    print(f"Silhouette Score: {silhouette_score(latent_matrix, labels):.2f}")

    # Clustering Quality
    kmeans = KMeans(n_clusters=len(le.classes_)).fit(latent_matrix)
    print(f"Adjusted Rand Index: {adjusted_rand_score(labels, kmeans.labels_):.2f}")