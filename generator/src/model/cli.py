import argparse
import torch
from .model import ModelArgs, SpatioTemporalTransformer, AutoEncoder
import src.transformations.transformations as transformations
import os


def main():
    parser = argparse.ArgumentParser(description='Initialize and test a model with custom parameters')
    
    parser.add_argument('--dim', type=int, default=64, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=64, help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--multiple-of', type=int, default=64, 
                       help='Make SwiGLU hidden layer size multiple of this value')
    parser.add_argument('--norm-eps', type=float, default=1e-5, help='RMSNorm epsilon value')
    parser.add_argument('--rope-theta', type=float, default=100.0, 
                       help='Theta for rotary positional embeddings')
    
    parser.add_argument('--out-channels', type=int, nargs='+', default=[16, 32, 64, 128],
                       help='Output channels for each conv layer')
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[3, 3, 3, 3],
                       help='Kernel sizes for conv layers')
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 1, 1, 1],
                       help='Strides for conv layers')
    parser.add_argument('--paddings', type=int, nargs='+', default=[1, 1, 1, 1],
                       help='Paddings for conv layers')
    parser.add_argument('--scaling-factor', type=int, default=2,
                       help='Scaling factor for max pooling')
    
    parser.add_argument('--model-type', choices=['transformer', 'autoencoder'], 
                       default='transformer', help='Type of model to create')
    
    parser.add_argument('--test', action='store_true', help='Run a test forward pass')
    parser.add_argument('--gif-path', type=str, help='Path to GIF for test input')

    args = parser.parse_args()

    conv_lists = [args.out_channels, args.kernel_sizes, args.strides, args.paddings]
    list_lengths = list(map(len, conv_lists))
    if len(set(list_lengths)) != 1:
        raise ValueError(
            f"Convolutional parameter lists must be of the same length. Got lengths: {list_lengths}"
        )

    if args.dim % args.n_heads != 0:
        raise ValueError(f"dim ({args.dim}) must be divisible by n_heads ({args.n_heads})")

    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        multiple_of=args.multiple_of,
        norm_eps=args.norm_eps,
        rope_theta=args.rope_theta,
        out_channel_sizes=args.out_channels,
        kernel_sizes=args.kernel_sizes,
        strides=args.strides,
        paddings=args.paddings,
        scaling_factor=args.scaling_factor
    )

    if args.model_type == 'transformer':
        model = SpatioTemporalTransformer(model_args)
    else:
        model = AutoEncoder(model_args)

    print(f"Created {args.model_type} model with parameters:")
    print(f"  dim: {args.dim}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  out_channels: {args.out_channels}")
    print(f"  kernel_sizes: {args.kernel_sizes}")

    if args.test:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        
        if args.gif_path:
            if not os.path.isfile(args.gif_path):
                raise FileNotFoundError(f"GIF file not found: {args.gif_path}")
            frames = transformations.transform_gif_to_tensor(args.gif_path)
        else:
            frames = torch.randn(1, 10, 3, 256, 256).to(device)  # (B, S, C, H, W)
        
        frames = transformations.transform_image_to_trainable_form(frames)
        print(f"\nInput shape: {frames.shape}")
        
        output = model(frames)
        print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()