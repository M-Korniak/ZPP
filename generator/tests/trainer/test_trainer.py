import os
import torch
import shutil
import pytest

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
def cleanup_dirs():
    # Setup: create directories if they don't exist
    os.makedirs("./autoencoder_save", exist_ok=True)
    os.makedirs("./transformer_save", exist_ok=True)
    yield
    # Teardown: remove directories after test
    shutil.rmtree("./autoencoder_save")
    shutil.rmtree("./transformer_save")

def test_save_and_load(cleanup_dirs):
    # Initialize the model with default args
    args = ModelArgs()
    autoencoder = model.AutoEncoder(args)
    transformer = model.SpatioTemporalTransformer(args)
    
    autoen_state_before, trans_state_before = autoencoder.state_dict(), transformer.state_dict()
   
    # Save the models
    save_model(autoencoder, args, "./autoencoder_save")
    save_model(transformer, args, "./transformer_save")

    # Load the models
    loaded_autoencoder = load_model("./autoencoder_save", "AutoEncoder", torch.device("cpu"))
    loaded_transformer = load_model("./transformer_save", "SpatioTemporalTransformer", torch.device("cpu"))

    # Verify state dictionaries
    autoen_state_after, trans_state_after = loaded_autoencoder.state_dict(), loaded_transformer.state_dict()
    assert compare_states(autoen_state_before, autoen_state_after)
    assert compare_states(trans_state_before, trans_state_after)

    # Verify hyperparameters
    # Extract args from the saved models
    autoencoder_checkpoint = torch.load("./autoencoder_save")
    transformer_checkpoint = torch.load("./transformer_save")

    loaded_autoencoder_args = autoencoder_checkpoint["hyperparams"]
    loaded_transformer_args = transformer_checkpoint["hyperparams"]
    
    assert compare_hyperparams(args, loaded_autoencoder_args)
    assert compare_hyperparams(args, loaded_transformer_args)

# Run the test
if __name__ == "__main__":
    pytest.main()