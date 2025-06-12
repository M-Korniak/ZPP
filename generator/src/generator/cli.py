import argparse
import torch
from src.generator.generator import generate_time_lapse_from_tensor
from src.transformations.transformations import unnormalize_image, transformations_for_evaluation
from src.data_processing.data_processing import get_dataloader
from src.visualizer.visualizer import visualize_tensor_images_as_gif
from src.trainer.trainer import load_model


def main():
    parser = argparse.ArgumentParser(description="Generate video using a pretrained model")
    parser.add_argument('--model-path', type=str, default='trained-models/saved_model64_200_alternative.pth', required=True, help='Path to the trained model')
    parser.add_argument('--model-type', type=str, choices=['transformer', 'autoencoder'], required=True, help='Model type')
    parser.add_argument('--data-folder', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--output-gif', type=str, required=True, help='Output path for the GIF')
    parser.add_argument('--video-length', type=int, default=258, help='Length of the generated video')
    parser.add_argument('--crop-size', type=int, default=16, help='Crop size used during transformation')
    parser.add_argument('--start-timestamp', type=int, default=100, help='Number of frames used as input for generation')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    model_type_str = 'SpatioTemporalTransformer' if args.model_type == 'transformer' else 'AutoEncoder'
    model = load_model(args.model_path, model_type_str, device)
    model.eval().to(device)

    _, test_loader = get_dataloader(
        data_folder=args.data_folder,
        batch_size=1,
        transform=lambda img: transformations_for_evaluation(img, crop_size=args.crop_size)
    )

    batch = next(iter(test_loader)).to(device)
    input_frames = batch[:, :args.start_timestamp]

    generated_video = generate_time_lapse_from_tensor(model, input_frames, video_length=args.video_length)
    generated_video = unnormalize_image(generated_video)

    visualize_tensor_images_as_gif(generated_video[0], path=args.output_gif)


if __name__ == "__main__":
    main()
