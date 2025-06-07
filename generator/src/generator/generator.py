import torch
from tqdm import tqdm


@torch.no_grad()
def generate_time_lapse_from_tensor(model: torch.nn.Module, input_frames: torch.tensor,
                               video_length: int = 258) -> torch.tensor:
    """
    Generates a video from a given tensor

    Args:
    - model (torch.nn.Module): The model to generate the video
    - input_frames (torch.tensor): The input frames to start the video
    - video_length (int): The length of the video to generate

    Returns:
    - torch.tensor: The generated video
    """
    B, S, C, H, W = input_frames.shape
    generated_frames = torch.zeros(B, video_length, C, H, W).to(input_frames.device)
    generated_frames[:, :S] = input_frames

    progress_bar = tqdm(range(S, video_length), desc="Generating rest of the video")

    for t in progress_bar:
        generated_frame = model(generated_frames[:, : t - 1])
        generated_frames[:, t] = generated_frame[:, -1]

    return generated_frames
