import torch
from tqdm import tqdm

def generate_video_from_tensor(model: torch.nn.Module, input_frames: torch.tensor,
                               video_length: int = 258) -> torch.tensor:
    B, S, C, H, W = input_frames.shape
    generated_frames = torch.zeros(B, video_length, C, H, W).to(input_frames.device)
    generated_frames[:, :S] = input_frames

    progress_bar = tqdm(range(S, video_length), desc="Generating rest of the video")

    for t in progress_bar:
        generated_frame = model(generated_frames[:, : t - 1])
        generated_frames[:, t] = generated_frame[:, -1]

    return generated_frames






