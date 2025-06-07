import torch
from unittest import mock
from src.generator.generator import generate_time_lapse_from_tensor

def test_generate_time_lapse_shape():
    B, S, C, H, W = 2, 10, 3, 64, 64
    video_length = 20

    input_tensor = torch.rand(B, S, C, H, W)

    mock_model = mock.MagicMock()
    mock_model.return_value = torch.rand(B, video_length, C, H, W)

    def mock_forward(input_seq):
        B, T, C, H, W = input_seq.shape
        last_frame = input_seq[:, -1].unsqueeze(1)
        return last_frame.repeat(1, 1, 1, 1, 1)

    mock_model.side_effect = mock_forward

    output = generate_time_lapse_from_tensor(mock_model, input_tensor, video_length)

    assert output.shape == (B, video_length, C, H, W)
    assert torch.allclose(output[:, :S], input_tensor, atol=1e-5)

def test_generate_time_lapse_no_growth():
    """Test that when video_length == S, input is returned as-is (no generation)."""
    B, S, C, H, W = 1, 5, 3, 32, 32
    input_tensor = torch.rand(B, S, C, H, W)

    mock_model = mock.MagicMock()

    output = generate_time_lapse_from_tensor(mock_model, input_tensor, video_length=S)

    assert output.shape == (B, S, C, H, W)
    assert torch.allclose(output, input_tensor, atol=1e-6)
    mock_model.assert_not_called()
