from src.generator.generator import Generator
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

@pytest.fixture
def sample_generator():
    """Fixture creating an instance of the Generator class with a sample DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 10),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 10),
        'other_column': np.random.randint(0, 5, 10)
    })
    return Generator(df)  # Tworzymy instancję klasy z przekazanym DataFrame

@pytest.fixture
def sample_generator_100_records():
    """Fixture creating an instance of the Generator class with a sample DataFrame holding 100 records."""
    np.random.seed(42)
    df = pd.DataFrame({
        'objNuclei_Location_Center_X': np.random.uniform(0, 256, 100),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 256, 100),
        'other_column': np.random.randint(0, 5, 100)
    })
    return Generator(df)  # Tworzymy instancję klasy z przekazanym DataFrame

@pytest.fixture
def sample_generator_extra_columns():
    """Fixture creating an instance of the Generator class with a sample DataFrame containing additional irrelevant columns."""
    np.random.seed(42)
    df = pd.DataFrame({
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 10),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 10),
        'other_column': np.random.randint(0, 5, 10),
        'irrelevant_column_1': np.random.choice(['A', 'B', 'C'], 10),  # Kolumna tekstowa
        'irrelevant_column_2': np.random.uniform(0, 1, 10),  # Kolumna numeryczna
        'irrelevant_column_3': np.random.randint(100, 200, 10)  # Kolejna liczba całkowita
    })
    return Generator(df)  # Tworzymy instancję klasy z rozszerzonym DataFrame

@pytest.fixture
def sample_data():
    """Fixture creating a sample DataFrame for tests."""
    return pd.DataFrame({
        'track_id': np.arange(5),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 5),
        'ERKKTR_ratio': np.random.uniform(0.5, 2.5, 5)
    })

@pytest.fixture
def adjacency_matrix():
    """Fixture for a sample adjacency matrix."""
    return torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=torch.float32, device=DEVICE)

@pytest.fixture
def disconnected_adjacency():
    """Adjacency matrix in which no nuclei are connected."""
    return torch.zeros((5, 5), dtype=torch.float32, device=DEVICE)

@pytest.fixture
def sample_points():
    """Fixture creating a sample DataFrame with nucleus positions."""
    return pd.DataFrame({
        'track_id': np.arange(5),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Z': np.random.uniform(0, 50, 5)
    })

@pytest.fixture
def simple_case():
    """Fixture for simple data where the neighborhood is obvious."""
    return pd.DataFrame({
        'track_id': [1, 2, 3, 4, 5],
        'objNuclei_Location_Center_X': [1, 3, 2, 0, 3], 
        'objNuclei_Location_Center_Y': [3, 3, 2, 0, 0]
    })

@pytest.mark.parametrize("num_points", [10, 50, 100])
def test_various_sizes(num_points, sample_generator):
    """Checks the function's behavior for different numbers of points."""
    np.random.seed(42)
    data = pd.DataFrame({
        'track_id': np.arange(num_points),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, num_points),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, num_points)
    })

    adjacency_matrix = sample_generator.calculate_neighbors(data)

    assert adjacency_matrix.shape == (num_points, num_points), f"Incorrect matrix size for {num_points} points."

@pytest.fixture
def sample_initial_frame():
    """Fixture: sample first frame of the simulation."""
    return pd.DataFrame({
        'Image_Metadata_T': np.zeros(5),
        'track_id': np.arange(5),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 5),
        'ERKKTR_ratio': np.random.uniform(0.5, 2.5, 5)
    })

def test_return_type(sample_generator):
    """Tests whether the function returns an object of type pd.DataFrame."""
    result = sample_generator.generate_next_move(sample_generator.df_first_frame.copy())
    assert isinstance(result, pd.DataFrame), "Return type should be pd.DataFrame"

def test_dataframe_shape(sample_generator):
    """Checks whether the number of rows and columns remains unchanged after calling the function."""
    original_shape = sample_generator.df_first_frame.shape
    result = sample_generator.generate_next_move(sample_generator.df_first_frame.copy())
    assert result.shape == original_shape, "The number of rows or columns has changed."

def test_x_y_columns_change(sample_generator):
    """Checks whether the values in columns X and Y change after calling the function."""
    original_x = sample_generator.df_first_frame['objNuclei_Location_Center_X'].copy()
    original_y = sample_generator.df_first_frame['objNuclei_Location_Center_Y'].copy()

    result = sample_generator.generate_next_move(sample_generator.df_first_frame.copy())

    assert not np.allclose(original_x, result['objNuclei_Location_Center_X']), "The X column has not changed its values."
    assert not np.allclose(original_y, result['objNuclei_Location_Center_Y']), "The Y column has not changed its values."

def test_other_columns_unchanged(sample_generator_extra_columns):
    """Checks whether other columns are not modified by the function."""

    irrelevant_column_1 = sample_generator_extra_columns.df_first_frame['irrelevant_column_1'].copy()
    irrelevant_column_2 = sample_generator_extra_columns.df_first_frame['irrelevant_column_2'].copy()
    irrelevant_column_3 = sample_generator_extra_columns.df_first_frame['irrelevant_column_3'].copy()
    
    result = sample_generator_extra_columns.generate_next_move(sample_generator_extra_columns.df_first_frame.copy())

    assert np.array_equal(irrelevant_column_1, result['irrelevant_column_1']), "Irrelevan column_1 has changed!"
    assert np.array_equal(irrelevant_column_2, result['irrelevant_column_2']), "Irrelevan column_2 has changed!"
    assert np.array_equal(irrelevant_column_3, result['irrelevant_column_3']), "Irrelevan column_3 has changed!"

def test_random_distribution(sample_generator_100_records):
    """Checks whether the differences in X and Y follow a normal distribution."""
    result = sample_generator_100_records.generate_next_move(sample_generator_100_records.df_first_frame.copy())

    diffs_x = result['objNuclei_Location_Center_X'] - sample_generator_100_records.df_first_frame['objNuclei_Location_Center_X']
    diffs_y = result['objNuclei_Location_Center_Y'] - sample_generator_100_records.df_first_frame['objNuclei_Location_Center_Y']

    assert np.isclose(np.mean(diffs_x), 0, atol=0.1), "The mean of differences in X deviates from the expected value."
    assert np.isclose(np.mean(diffs_y), 0, atol=0.1), "The mean of differences in Y deviates from the expected value."
    assert np.isclose(np.std(diffs_x), 0.43, atol=0.1), "The deviation in X deviates from the expected value."
    assert np.isclose(np.std(diffs_y), 0.43, atol=0.1), "The deviation in Y deviates from the expected value."

def test_output_dataframe_shape(adjacency_matrix, sample_data, sample_generator):
    """Checks whether the number of rows in the returned DataFrame is correct."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)
    
    assert isinstance(result, pd.DataFrame), "The function should return a DataFrame."
    assert result.shape[0] == sample_data.shape[0], "The number of rows in the DataFrame should remain the same."

def test_output_dataframe_shape_after_5_iterations(adjacency_matrix, sample_data, sample_generator):
    """Checks whether the number of rows in the returned DataFrame after five iterations is correct."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=5)
    
    assert isinstance(result, pd.DataFrame), "The function should return a DataFrame."
    assert result.shape[0] == sample_data.shape[0], "The number of rows in the DataFrame should remain the same."

def test_ERKKTR_ratio_range(sample_data, adjacency_matrix, sample_generator):
    """Checks whether the ERKKTR_ratio values are constrained to the range [0.4, 2.7]."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)
    
    assert result['ERKKTR_ratio'].between(0.4, 2.7).all(), "ERKKTR_ratio values should be constrained to [0.4, 2.7]."

def test_ERKKTR_ratio_range_after_100_iterations(sample_data, adjacency_matrix, sample_generator):
    """Checks whether the ERKKTR_ratio values are constrained to the range [0.4, 2.7] after 100 iterations."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=100)
    
    assert result['ERKKTR_ratio'].between(0.4, 2.7).all(), "ERKKTR_ratio values should be constrained to [0.4, 2.7]."

def test_track_id_sorted(sample_data, adjacency_matrix, sample_generator):
    """Checks whether the results are sorted by track_id."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)
    
    assert (result['track_id'].values == sorted(result['track_id'].values)).all(), "track_id should be sorted."

def test_ERKKTR_ratio_changes(sample_data, adjacency_matrix, sample_generator):
    """Checks whether the ERKKTR_ratio values change after applying the function."""
    original_values = sample_data['ERKKTR_ratio'].copy()
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)

    assert not np.allclose(original_values, result['ERKKTR_ratio']), "ERKKTR_ratio should change."

def test_disconnected_graph(sample_data, disconnected_adjacency, sample_generator):
    """Checks whether the ERKKTR_ratio values remain unchanged when there are no neighbors."""
    original_values = sample_data['ERKKTR_ratio'].copy()
    result = sample_generator.generate_next_ERK(sample_data, disconnected_adjacency, T=1)

    assert np.allclose(original_values, result['ERKKTR_ratio'], atol=0.1), "ERKKTR_ratio should not change significantly."

def test_output_tensor_shape(sample_points, sample_generator):
    """Checks whether the returned adjacency matrix has the correct dimensions."""
    result = sample_generator.calculate_neighbors(sample_points)

    assert isinstance(result, torch.Tensor), "Function should return torch.Tensor"
    assert result.shape == (len(sample_points), len(sample_points)), "The adjacency matrix should have a size of NxN."

def test_adjacency_symmetry(sample_points, sample_generator):
    """Checks whether the adjacency matrix is symmetric."""
    adjacency_matrix = sample_generator.calculate_neighbors(sample_points)

    assert torch.equal(adjacency_matrix, adjacency_matrix.T), "The matrix should be symmetric."

def test_no_self_connections(sample_points, sample_generator):
    """Checks whether the adjacency matrix has no self-connections."""
    adjacency_matrix = sample_generator.calculate_neighbors(sample_points)

    assert torch.all(torch.diag(adjacency_matrix) == 0), "There should be no self-connections."

def test_simple_neighbors(simple_case, sample_generator):
    """Checks whether the neighborhood for ordered points is correct."""
    adjacency_matrix = sample_generator.calculate_neighbors(simple_case)
    expected = torch.tensor([[0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0]
    ], dtype=torch.uint8, device=DEVICE)

    assert torch.equal(adjacency_matrix, expected), "The adjacency matrix should be correct for a simple case."

def test_generate_video_returns_dataframe(sample_initial_frame, sample_generator):
    """Checks whether the generate_video function returns a DataFrame."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    # Mockowanie funkcji pomocniczych
    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    assert isinstance(result, pd.DataFrame), "Function should return `pd.DataFrame`"

def test_generate_video_columns(sample_initial_frame, sample_generator):
    """Checks whether the returned DataFrame contains all required columns."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    expected_columns = {'track_id', 'objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y', 
                        'ERKKTR_ratio'}
    assert expected_columns.issubset(result.columns), "Required columns are missing in the result."

def test_generate_video_frame_count(sample_initial_frame, sample_generator):
    """Checks whether the resulting DataFrame contains the correct number of frames."""

    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    num_tracks = len(sample_initial_frame)
    expected_rows = num_tracks * sample_generator.number_of_frames
    assert len(result) == expected_rows, f"Incorrect number of rows: {len(result)}, expected {expected_rows}."

def test_generate_video_track_id_integrity(sample_initial_frame, sample_generator):
    """Checks whether track_id identifiers remain constant over time."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    grouped = result.groupby("track_id")
    assert all(len(group) == sample_generator.number_of_frames for _, group in grouped), "Not all track_id have the correct number of frames."

def test_generate_video_time_metadata(sample_initial_frame, sample_generator):
    """Checks whether Image_Metadata_T is correctly incremented over time."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    unique_times = sorted(result["Image_Metadata_T"].unique())
    assert unique_times == list(range(sample_generator.number_of_frames)), "Incorrect Image_Metadata_T values."

def test_generate_video_changes_values(sample_initial_frame, sample_generator):
    """Checks whether ERKKTR_ratio and the X, Y coordinates actually change over time."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    def fake_next_move(df):
        df = df.copy()
        df["objNuclei_Location_Center_X"] += np.random.normal(0, 1, len(df))
        df["objNuclei_Location_Center_Y"] += np.random.normal(0, 1, len(df))
        return df

    def fake_next_ERK(df, adj_matrix, T):
        df = df.copy()
        df["ERKKTR_ratio"] += np.random.normal(0, 0.1, len(df))
        return df

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = fake_next_ERK
    sample_generator.generate_next_move = fake_next_move

    result = sample_generator.generate_video()

    assert result["objNuclei_Location_Center_X"].nunique() > len(sample_initial_frame), "X values do not change."
    assert result["objNuclei_Location_Center_Y"].nunique() > len(sample_initial_frame), "Y values do not change."
    assert result["ERKKTR_ratio"].nunique() > len(sample_initial_frame), "ERK does not change over time."
