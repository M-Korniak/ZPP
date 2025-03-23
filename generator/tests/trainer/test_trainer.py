import os
import torch
import shutil
import pytest
import tempfile  # For creating temporary directories

import src.model.model as model
from src.model.model import ModelArgs
from src.trainer.trainer import save_model, load_model

def compare_states(state1, state2):
    """
    Compare two state dictionaries using torch.equal.
    """
    if state1.keys() != state2.keys():
        print("Keys do not match!")
        return False

    # Compare all tensors in the state dictionaries
    for key in state1.keys():
        if not torch.equal(state1[key], state2[key]):
            print(f"Mismatch in key: {key}")
            return False

    print("State dictionaries match!")
    return True

def compare_hyperparams(args1, args2):
    """
    Compare two sets of hyperparameters.
    """
    return args1 == args2

@pytest.fixture
def temp_dirs():
    """
    Fixture to create temporary directories for testing.
    """
    # Create temporary directories
    with tempfile.TemporaryDirectory() as autoencoder_dir, tempfile.TemporaryDirectory() as transformer_dir:
        yield autoencoder_dir, transformer_dir  # Pass the paths to the test
        # Directories are automatically cleaned up after the test

def test_save_and_load(temp_dirs):
    """
    Test saving and loading of AutoEncoder and SpatioTemporalTransformer models.
    """
    autoencoder_dir, transformer_dir = temp_dirs

    # Initialize the model with default args
    args = ModelArgs()
    autoencoder = model.AutoEncoder(args)
    transformer = model.SpatioTemporalTransformer(args)
    
    # Save the state dictionaries before saving the models
    autoen_state_before = autoencoder.state_dict()
    trans_state_before = transformer.state_dict()
   
    # Save the models
    save_model(autoencoder, args, os.path.join(autoencoder_dir, "autoencoder_save"))
    save_model(transformer, args, os.path.join(transformer_dir, "transformer_save"))

    # Load the models
    loaded_autoencoder = load_model(os.path.join(autoencoder_dir, "autoencoder_save"), "AutoEncoder", torch.device("cpu"))
    loaded_transformer = load_model(os.path.join(transformer_dir, "transformer_save"), "SpatioTemporalTransformer", torch.device("cpu"))

    # Verify state dictionaries
    autoen_state_after = loaded_autoencoder.state_dict()
    trans_state_after = loaded_transformer.state_dict()
    assert compare_states(autoen_state_before, autoen_state_after)
    assert compare_states(trans_state_before, trans_state_after)

    # Verify hyperparameters
    # Extract args from the saved models
    autoencoder_checkpoint = torch.load(os.path.join(autoencoder_dir, "autoencoder_save"))
    transformer_checkpoint = torch.load(os.path.join(transformer_dir, "transformer_save"))

    loaded_autoencoder_args = autoencoder_checkpoint["hyperparams"]
    loaded_transformer_args = transformer_checkpoint["hyperparams"]
    
    assert compare_hyperparams(args, loaded_autoencoder_args)
    assert compare_hyperparams(args, loaded_transformer_args)