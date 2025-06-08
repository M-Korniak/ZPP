import argparse
import torch
from src.trainer.trainer import load_model

def main():
    parser = argparse.ArgumentParser(description="Load a trained model and set it to evaluation mode.")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the saved model file (e.g., model.pth)")
    parser.add_argument("--model-type", type=str, choices=["SpatioTemporalTransformer", "AutoEncoder"],
                        default="SpatioTemporalTransformer", help="Type of model to load")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default="cuda",
                        help="Device to load the model on")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.model_path, args.model_type, device)
    model.eval()

    print(f"Model '{args.model_type}' loaded from {args.model_path} and set to eval mode on {device}.")

if __name__ == "__main__":
    main()
