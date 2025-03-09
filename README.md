# Biomedical Data Generation Package

This package was developed as part of a bachelor's thesis at the Faculty of Mathematics, Informatics, and Mechanics at the University of Warsaw. It enables the generation of biomedical data based on provided experimental data. By leveraging a transformer model and a heuristic generator, the package can produce data resembling real experimental results.

The input data must be supplied in DataFrames with specific column names, ensuring consistency in processing. The package offers three main pipelines:

1. **Heuristic Data Generation** – Generates synthetic data based on predefined rules.  
2. **Transformer-Based Model Training** – Trains a transformer model to predict the next frame in a dataset.  
3. **Preprocessing Pipeline** – Converts DataFrames into tensors suitable for model training.

## Installation

To install the package, use the following command:

```bash
pip install biomedical-data-gen
```
Alternatively, install from the source.
```bash
git clone https://github.com/your-repo/biomedical-data-gen.git
cd biomedical-data-gen
pip install -e .
```
# Using the Heuristic Data Generator

## Prerequisites

To use the heuristic data generator, you need to provide an initial dataset (`df_first_frame`) in the form of a Pandas DataFrame. The DataFrame must contain the following columns:

- `track_id`: A unique identifier for each tracked nucleus.
- `objNuclei_Location_Center_X`: The X-coordinate of the nucleus in the frame.
- `objNuclei_Location_Center_Y`: The Y-coordinate of the nucleus in the frame.
- `ERKKTR_ratio`: The ERK activity level for the nucleus.
- `Image_Metadata_T`: The frame index (starting from `0` for the initial frame).

### Example of Required DataFrame Structure

| track_id | objNuclei_Location_Center_X | objNuclei_Location_Center_Y | ERKKTR_ratio | Image_Metadata_T |
|----------|-----------------------------|-----------------------------|--------------|------------------|
| 1        | 50.2                        | 120.4                       | 1.2          | 0                |
| 2        | 55.3                        | 118.1                       | 1.4          | 0                |
| 3        | 60.1                        | 122.5                       | 1.1          | 0                |

The generator will use this data as the starting point and create new frames by simulating the movement of nuclei and changes in ERK levels.

## How to Use

### Step 1: Load Data

Before running the generator, load your initial dataset (`df_first_frame`). The dataset should be a Pandas DataFrame with specific column names.

```python
df = utils.unpack_and_read("path/to/data.csv.gz")

df_first_frame = df[
    (df["Image_Metadata_Site"] == 1) & 
    (df["Exp_ID"] == 1) & 
    (df["Image_Metadata_T"] == 0)
][["track_id", "objNuclei_Location_Center_X", "objNuclei_Location_Center_Y", "ERKKTR_ratio", "Image_Metadata_T"]]
```

### Step 2: Initialize and Run the Generator

Once the initial dataset is loaded, initialize the Generator class with the first frame and specify the number of frames to simulate (default is 258). Then, generate the synthetic video data.

```python
generator = Generator(df_first_frame=df_first_frame)
video_data = generator.generate_video()
```
This will:

Simulate the movement of nuclei frame by frame.
Update the ERK values based on neighboring interactions.
Generate a final dataset containing all frames.

### Step 3: Visualize the Generated Data
To inspect the generated simulation, use the built-in visualizer.
```python
visualizer.visualize_simulation(video_data)
```
The visualization will display how the nuclei move and how ERK values evolve over time.

### Expected Output

The output (`video_data`) is a Pandas DataFrame containing all generated frames.

#### Example Output Structure:

| track_id | objNuclei_Location_Center_X | objNuclei_Location_Center_Y | ERKKTR_ratio | Image_Metadata_T |
|----------|-----------------------------|-----------------------------|--------------|------------------|
| 1        | 50.5                        | 120.1                       | 1.25         | 1                |
| 2        | 55.1                        | 118.5                       | 1.38         | 1                |
| 3        | 60.3                        | 122.7                       | 1.15         | 1                |
| ...      | ...                          | ...                         | ...          | ...              |

Each row represents a nucleus at a specific time frame, with updated positions and ERK levels.

This approach allows for realistic simulation of cell movements and biochemical activity, making it useful for machine learning applications and hypothesis testing.

## Preprocessing Pipeline

The preprocessing pipeline is responsible for converting raw experimental data (stored in Pandas DataFrames) into tensors that can be used for model training. This process involves the following steps:

1. **Loading and Cleaning Data** – The raw experimental dataset is loaded, and outliers in `ERKKTR_ratio` are clipped to a specific range.
2. **Generating Visual Representations** – The tabular data is transformed into visual representations (GIFs) for each field of view.
3. **Converting GIFs to Tensors** – The generated images are converted into tensor format, ensuring they have a consistent shape.
4. **Saving Processed Data** – The tensors are saved in a structured format to be used later in model training.
5. **Creating a Dataset and DataLoader** – The processed tensors are stored in a dataset and can be loaded efficiently using PyTorch's `DataLoader`.

---

### How It Works

The function `load_experiment_data_to_tensor()` processes the raw dataset and converts it into tensors.

```python
load_experiment_data_to_tensor(experiments=(1, 2, 3), maintain_experiment_visualization=False)
```
### What Happens

1. **Loading Data**: 
   - The function reads the dataset from a CSV file located at `../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz`.
   - `ERKKTR_ratio` values are clipped to the range `[0.4, 2.7]` to avoid extreme outliers.

2. **Processing Each Experiment**:
   - For each experiment (defined in the `experiments` tuple):
     - The relevant data is filtered from the full dataset based on the experiment ID.
     - A sorted list of unique field of view (FOV) IDs is created for the experiment.

3. **Visualizing and Converting Data**:
   - For each FOV in the experiment:
     - The number of frames (`frames_count`) for that field of view is determined.
     - A GIF visualization of the simulation is generated using `visualizer.visualize_simulation()`.
     - The GIF is then converted into a tensor using `transformations.transform_gif_to_tensor()`.
     - If the tensor has fewer than 258 frames, it is padded with zeros to make it consistent.

4. **Saving Tensors**:
   - After processing the FOV data:
     - The generated tensor is saved using `torch.save()` to the directory `../../data/tensors_to_load/`.
     - The corresponding GIF file is removed if `maintain_experiment_visualization` is `False`.

5. **Clean-up**:
   - If `maintain_experiment_visualization` is `False`, the temporary folder `../../data/experiments` is removed to clean up the generated GIF files.

---

### Example Usage

#### Step 1: Convert DataFrames to Tensors
To start the preprocessing, run the following code to process the data:

```python
load_experiment_data_to_tensor(experiments=(1, 2, 3))
```
### Step 2: Load Data from Saved Tensors

After preprocessing the data in Step 1, you can now load the saved tensors for further use in your models or for analysis.

The `TensorDataset` class is used to load the saved tensor files and prepare them for training or inference. Here's how to load the data:

#### Example Usage

```python
# Initialize the TensorDataset by pointing to the folder containing the tensor files
dataset = TensorDataset("../../data/tensors_to_load/")

# Access the first tensor in the dataset
sample_tensor = dataset[0]  # Gets the first tensor from the dataset
```
### Step 3: Create DataLoader for Training and Testing

Now that the dataset is ready, the next step is to create `DataLoader` objects to load the data in batches for training and testing. `DataLoader` provides an efficient way to load and shuffle the data.

#### Example Usage

```python
# Get the train and test DataLoaders
train_dataloader, test_dataloader = get_dataloader(
    data_folder="../../data/tensors_to_load/",  # Path to the folder with tensor files
    load_to_ram=False,  # Load data lazily from disk (you can also load all into RAM if you want)
    batch_size=16,  # Size of each batch
    num_workers=4,  # Number of worker threads to load data in parallel
    pin_memory=True,  # Whether to pin memory (for faster data transfer to GPU)
    train_split=0.8,  # Fraction of data to use for training (80%)
    seed=42,  # Random seed for reproducibility
)
```
### Step 4: Visualize a Sample from the Dataset

After loading the tensor data, it's useful to visualize a sample to ensure that everything has been correctly preprocessed and is ready for use in training. This step allows you to check the data format and content before proceeding with model training or inference.

#### Example Usage

```python
# Load a tensor sample from the dataset
my_tensor = torch.load("../../data/tensors_to_load/experiments_tensor_exp_1_fov_1.pt")

# Visualize the first frame of the first experiment and field of view
visualizer.visualize_tensor_image(my_tensor[0][0])
```

# SpatioTemporal Transformer - Next Frame Generation

## Description

The SpatioTemporal Transformer is a model that uses convolutional networks (for extracting features from images) and a transformer mechanism (for analyzing temporal dependencies) to generate the next frames in a sequence. The model combines spatial analysis of images with temporal analysis, enabling it to predict future frames in a video or image sequence.

## Usage Example
To use the `SpatioTemporalTransformer` model to generate the next frame of a video based on previous frames, follow these steps:

### 1. Prepare the Input Data
Assuming you have input data in the form of a sequence of video frames in GIF format (or other image formats suitable for training):

```python
import transformations

# Convert GIF file to tensor
frames = transformations.transform_gif_to_tensor()

# Prepare images in a trainable format
frames = transformations.transform_image_to_trainable_form(frames)

print(frames.shape)
```

### 2. Initialize the Model

Once you have prepared your input data, the next step is to initialize the `SpatioTemporalTransformer` model. To do this, you need to define the model parameters and then create the model instance. Here's how you can do that:

```python
from model import SpatioTemporalTransformer, ModelArgs

# Initialize the model arguments
args = ModelArgs()

# Create the SpatioTemporalTransformer model
model = SpatioTemporalTransformer(args)

# Pass the prepared input data (frames) to the model
output = model(frames)

# Print the shape of the generated output to verify
print(output.shape)
```

### 3. Generate the Next Frame

After initializing the model and passing the input frames to it, the next step is to generate the subsequent frame. This is done by feeding the prepared input tensor to the model, which processes the data and outputs the next frame in the sequence. 

The generated output will be in the form of a tensor that represents the next frame of the video.

Here’s how you generate the next frame:

```python
# Generate the next frame by passing the input tensor to the model
next_frame = model(frames)

# Print the shape of the generated next frame to verify
print(next_frame.shape)

# Optionally, transform the output back into an image or frame format
# Example transformation (depends on your implementation):
next_frame_image = transformations.tensor_to_image(next_frame)

# Display or save the generated frame
# For example, display the frame using a library like PIL or OpenCV
from PIL import Image
Image.fromarray(next_frame_image).show()
```

## Contributing
Feel free to open issues or submit pull requests to improve the package.
## License
This project is licensed under the MIT License. (Not sure, about it)