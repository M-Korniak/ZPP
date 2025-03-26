import os
import pandas as pd
import numpy as np
import torch
   
from umap import UMAP 
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
from src.transformations.transformations import transformations_for_evaluation

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

def get_latent_representations(model: torch.nn.Module, 
                              input_tensors: dict,
                              metadata: dict,
                              crop_size: int = 64) -> tuple:
    """
    Collect latent representations from both encoder and transformer spaces.
    
    Args:
        model: Pretrained model with get_encoder_latent_space and get_encoder_transformer_space methods
        input_tensors: Nested dictionary of input tensors
        metadata: Experiment metadata dictionary
        crop_size: Size for center cropping
        
    Returns:
        Tuple of (latent_matrix, transformer_matrix, cell_types)
    """
    latent_vectors = []
    transformer_vectors = []
    cell_types = []
    
    for exp_id in input_tensors:
        for fov in input_tensors[exp_id]:
            if exp_id not in metadata or fov not in metadata[exp_id]:
                continue
                
            # Process tensor
            tensor = input_tensors[exp_id][fov]
            tensor_for_eval = transformations_for_evaluation(tensor, crop_size=crop_size)
            
            with torch.no_grad():
                latent = model.get_encoder_latent_space(tensor_for_eval).flatten().cpu().numpy()
                transformer = model.get_encoder_transformer_space(tensor_for_eval).flatten().cpu().numpy()
                
            latent_vectors.append(latent)
            transformer_vectors.append(transformer)
            cell_types.append(metadata[exp_id][fov])
    
    return np.array(latent_vectors), np.array(transformer_vectors), cell_types

def visualize_pca(matrix: np.ndarray, 
                 cell_types: list, 
                 n_components: int = 2, 
                 palette: str = 'tab10',
                 title: str = 'PCA Visualization') -> pd.DataFrame:
    """
    Perform PCA and visualize the first two principal components.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(matrix)

    df_pca = pd.DataFrame(pca_result, 
                         columns=[f'PC{i+1}' for i in range(n_components)])
    df_pca['Cell_Type'] = cell_types

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cell_Type',
                   data=df_pca, palette=palette, s=100,
                   edgecolor='black', alpha=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return df_pca

def visualize_umap(matrix: np.ndarray, 
                  cell_types: list, 
                  palette: str = 'tab10',
                  title: str = 'UMAP Visualization') -> pd.DataFrame:
    """
    Perform UMAP dimensionality reduction and visualize results.
    """
    reducer = UMAP(random_state=42) # for reproducibility
    umap_result = reducer.fit_transform(matrix)

    df_umap = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    df_umap['Cell_Type'] = cell_types

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cell_Type',
                   data=df_umap, palette=palette, s=100,
                   edgecolor='black', alpha=0.8)
    plt.title(title, fontsize=14)
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return df_umap

# Additional evaluation
def evaluate_classification(matrix: np.ndarray, 
                          cell_types: list, 
                          test_size: float = 0.2) -> float:
    """
    Evaluate classification accuracy using logistic regression.

    Args:
        matrix: Input data matrix
        cell_types: List of cell type labels
        test_size: Proportion of data to use for testing

    Returns:
        Classification accuracy score
    """
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, cell_types, test_size=test_size, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Classification Accuracy: {accuracy:.3f}")
    return accuracy

def evaluate_silhouette(matrix: np.ndarray, 
                       cell_types: list) -> float:
    """
    Calculate silhouette score for clustering quality.

    Args:
        matrix: Input data matrix
        cell_types: List of cell type labels

    Returns:
        Silhouette score
    """
    le = LabelEncoder()
    labels = le.fit_transform(cell_types)
    score = silhouette_score(matrix, labels)
    print(f"Silhouette Score: {score:.3f}")
    return score

def evaluate_clustering(matrix: np.ndarray, 
                       cell_types: list) -> float:
    """
    Evaluate clustering performance using Adjusted Rand Index.

    Args:
        matrix: Input data matrix
        cell_types: List of cell type labels

    Returns:
        Adjusted Rand Index score
    """
    le = LabelEncoder()
    true_labels = le.fit_transform(cell_types)
    kmeans = KMeans(n_clusters=len(le.classes_), random_state=42)
    pred_labels = kmeans.fit_predict(matrix)
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"Adjusted Rand Index: {ari:.3f}")
    return ari


# Latent space analysis
if __name__ == "__main__":
    # Example usage
    metadata_path = "../../data/plateMap.csv"
    input_data_path = "../../data/tensors_to_load"
    path_to_model = "../../data/saved_model64_200_alternative.pth"
    
    # Uncomment if you don't have the files with tensors in input_data_path yet
    # load_experiment_data_to_tensor()
    
    # Loading experiments metadata
    extracted_metadata = load_metadata(metadata_path)
    print("Extracting metadata finished.")

    input_tensors = load_pt_files(input_data_path)
    print("Getting the input tensors finished.")
    
    # Loading the trained model
    trained_model = load_model(path_to_model, 'SpatioTemporalTransformer', torch.device("cpu"))
    print("Loading the pre-trained model finished.")
    trained_model.eval()
    
    # Get latent representations
    latent_vectors = []
    cell_types = []
    exp_ids_list = []

    latent_matrix, transformer_matrix, cell_types = get_latent_representations(
        trained_model, input_tensors, extracted_metadata
    )

    # Visualization
    print("\nLatent Space Analysis:")
    latent_pca = visualize_pca(latent_matrix, cell_types,
                             n_components=5, 
                             title='PCA of Encoder Latent Space')
    latent_umap = visualize_umap(latent_matrix, cell_types,
                               title='UMAP of Encoder Latent Space')

    print("\nTransformer Space Analysis:")
    transformer_pca = visualize_pca(transformer_matrix, cell_types,
                                  n_components=5,
                                  title='PCA of Transformer Space')
    transformer_umap = visualize_umap(transformer_matrix, cell_types,
                                    title='UMAP of Transformer Space')

    # Quantitative evaluation
    print("\nLatent Space Performance:")
    evaluate_classification(latent_matrix, cell_types)
    evaluate_silhouette(latent_matrix, cell_types)
    evaluate_clustering(latent_matrix, cell_types)

    print("\nTransformer Space Performance:")
    evaluate_classification(transformer_matrix, cell_types)
    evaluate_silhouette(transformer_matrix, cell_types)
    evaluate_clustering(transformer_matrix, cell_types)

    # TODO: add t-SNE
    # TODO: try out single frames instead of the videos
    # TODO: add the option to save the visualisations