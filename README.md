# Biomedical Data Generation Package

This package was developed as part of a bachelor's thesis at the Faculty of Mathematics, Informatics, and Mechanics at the University of Warsaw. It enables the generation, preprocessing, and modeling of biomedical data from time-lapse microscopy experiments. It supports both rule-based (heuristic) simulation and training of the SpatioTemporalTransformer model for predictive video generation.

It can be used either programmatically or via a command-line interface (CLI), with full support for integration with ClearML, automatic preprocessing, and visualization

The input data must be supplied in DataFrames with specific column names, ensuring consistency in processing. The package offers three main pipelines:

1. **Heuristic Data Generation** – Generates synthetic data based on predefined rules.  
2. **Data-Based Model Training** – Trains a transformer model to predict the next frame in a dataset.  
3. **Preprocessing Pipeline** – Converts DataFrames into tensors suitable for model training.

## Installation

To install the package, use the following command:

```bash
pip install modelcellsignaling
```
Alternatively, install from the source.
```bash
git clone https://github.com/your-repo/biomedical-data-gen.git
cd ZPP/generator
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
from src.rule_based_generator.rule_based_generator import RuleBasedGenerator
import src.utils.utils
import src.visualizer.visualizer

df = utils.unpack_and_read("data.csv.gz")
df_first_frame = df[(df['Image_Metadata_Site'] == 1) & (df['Exp_ID'] == 1) & (df['Image_Metadata_T'] == 0)][[
    'track_id', 'objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y', 'ERKKTR_ratio', 'Image_Metadata_T']]

```

### Step 2: Initialize and Run the Generator

Once the initial dataset is loaded, initialize the Generator class with the first frame and specify the number of frames to simulate (default is 258). Then, generate the synthetic video data.

```python
generator = RuleBasedGenerator(df_first_frame=df_first_frame, mutation_type='PTEN_del')
simulated_df = generator.generate_time_lapse()
```
This will:

Simulate the movement of nuclei frame by frame.
Update the ERK values based on neighboring interactions.
Generate a final dataset containing all frames.

### Step 3: Visualize the Generated Data
To inspect the generated simulation, use the built-in visualizer.
```python
visualizer.visualize_simulation(simulated_df)
```
The visualization will display how the nuclei move and how ERK values evolve over time.

### Alternative CLI Usage
```python
generate-time-lapse-rule-based \
    --input data.csv.gz \
    --exp-id 1 \
    --site 1 \
    --frames 258 \
    --mutation-type WT \
    --output ./simulated_time_lapse.csv \
    --visualize \
    --save-path
```

### Supported Mutation Types

The following mutation types are supported by the rule-based generator:

- **WT** – Wild Type (no mutation)
- **AKT1_E17K** – Mutation in AKT1 gene at position E17K
- **PIK3CA_E545K** – Mutation in PIK3CA gene at position E545K
- **PIK3CA_H1047R** – Mutation in PIK3CA gene at position H1047R
- **PTEN_del** – Deletion of the PTEN gene

### Expected Output

The output is a Pandas DataFrame containing all generated frames.

#### Example Output Structure:

| track_id | objNuclei_Location_Center_X | objNuclei_Location_Center_Y | ERKKTR_ratio | Image_Metadata_T |
|----------|-----------------------------|-----------------------------|--------------|------------------|
| 1        | 50.5                        | 120.1                       | 1.25         | 1                |
| 2        | 55.1                        | 118.5                       | 1.38         | 1                |
| 3        | 60.3                        | 122.7                       | 1.15         | 1                |
| ...      | ...                         | ...                         | ...          | ...              |

Each row represents a nucleus at a specific time frame, with updated positions and ERK levels.

This approach allows for realistic simulation of cell movements and biochemical activity, making it useful for machine learning applications and hypothesis testing.

## Preprocessing Pipeline

The preprocessing pipeline performs the following steps:

- Loads tabular CSV data and filters it by experiment.
- Clips outlier `ERKKTR_ratio` values to a predefined range.
- Generates GIF visualizations for each field of view (FOV).
- Converts GIFs to PyTorch tensors with consistent shape.
- Saves the resulting tensor files to disk for model training.

---

### How It Works

The function `load_experiment_data_to_tensor()` processes the raw dataset and converts it into tensors.

```python
load_experiment_data_to_tensor(experiments=(1, 2, 3))
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

### Alternative CLI Usage
```python
process-tensor \
    --load-data \
    --data-path ./data.csv.gz \
    --tensor-path ./data/tensors_to_load

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

# SpatioTemporalTransformer Usage

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
from src.model.model import SpatioTemporalTransformer, ModelArgs
from src.trainer.trainer import Trainer, save_model
import src.transformations.transformations as tf

args = ModelArgs()
model = SpatioTemporalTransformer(args).cuda()
trainer = Trainer(n_epochs=100, batch_size=4,
                  extra_augmentation=tf.transformations_for_training)
trainer.train(model)
save_model(model, args, "model.pth")

```

### 3. Generate the Next Frame

After initializing the model and passing the input frames to it, the next step is to generate the subsequent frame. This is done by feeding the prepared input tensor to the model, which processes the data and outputs the next frame in the sequence. 

The generated output will be in the form of a tensor that represents the next frames of the video.

Here’s how you generate the next frames:

```python
train_loader, test_loader = data_processing.get_dataloader(
    data_folder=args.data_folder,
    batch_size=1,
    transform=lambda image: transformations.transformations_for_evaluation(image, crop_size=args.crop_size)
)

model.eval().to(device)
batch = next(iter(test_loader)).to(device)
generated_time_lapse = generator.generate_time_lapse_from_tensor(model, batch[:, :100], video_length=258)
generated_time_lapse = transformations.unnormalize_image(generated_video)
visualizer.visualize_tensor_images_as_gif(generated_video[0], path=args.generate_gif)

```
### Alternative CLI Usage
```python
train-model \
    --model-type transformer \
    --epochs 200 \
    --lr 0.002 \
    --batch-size 4 \
    --crop-size 16 \
    --save-model ./model.pth \
    --generate-gif ./example_output.gif

```
# Load Pretrained Model
There is also a possibility to load pretrainded model. It can be done via CLI:
```python
load-model \
    --model-path model.pth \
    --model-type SpatioTemporalTransformer \
    --device cuda

```
and via Python:
```python
from src.trainer.trainer import load_model

model = load_model("model.pth", "SpatioTemporalTransformer", 
                   torch.device("cuda"))
model.eval()
```

# Autoregresive generation
After preprocessing data, and training the model, autoregressive generation can be performed:
```python
generate-time-lapse-autoregressive \
  --model-path models/checkpoint.pth \
  --model-type transformer \
  --data-folder ./data/test_dataset \
  --output-gif ./results/generated_video.gif \
  --video-length 150 \
  --crop-size 16
  --start-timestamp 100
```

## Contributing
Feel free to open issues or submit pull requests to improve the package.
## License
This project is licensed under the MIT License. (Not sure, about it)