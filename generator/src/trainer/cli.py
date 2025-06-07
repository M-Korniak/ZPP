import argparse
import torch
from .trainer import Trainer, AutoEncoderTrainer
import src.model.model as model_module  
import src.visualizer.visualizer as visualizer
import src.generator.generator as generator
import src.transformations.transformations as transformations
from .trainer import load_model, save_model 
import src.data_processing.data_processing as data_processing


def train_model():
    parser = argparse.ArgumentParser(description='Train SpatioTemporal or AutoEncoder model')
    
    parser.add_argument('--model-type', type=str, choices=['transformer', 'autoencoder'], 
                       default='transformer', help='Type of model to train')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--weight-decay', type=float, default=3e-5, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.01, help='Batch norm momentum')
    
    parser.add_argument('--data-folder', type=str, default='./data/tensors_to_load',
                       help='Path to folder with training tensors')
    parser.add_argument('--load-to-ram', action='store_true', 
                       help='Load all data to RAM for faster training')
    parser.add_argument('--crop-size', type=int, default=16,
                       help='Size of random crops for data augmentation')
    
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save trained model (optional)')
    parser.add_argument('--generate-gif', type=str, default=None,
                       help='Path to save generated GIF (optional)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load existing model (optional)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    if args.load_model:
        model_type = 'SpatioTemporalTransformer' if args.model_type == 'transformer' else 'AutoEncoder'
        model = load_model(args.load_model, model_type, device)
    else:
        model_args = model_module.ModelArgs()  
        model = (model_module.SpatioTemporalTransformer(model_args) if args.model_type == 'transformer' 
                else model_module.AutoEncoder(model_args)).to(device)

    trainer_class = Trainer if args.model_type == 'transformer' else AutoEncoderTrainer
    trainer = trainer_class(
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        load_to_ram=args.load_to_ram,
        batch_norm_momentum=args.momentum,
        extra_augmentation=lambda image: transformations.transformations_for_training(image, crop_size=args.crop_size),
        data_folder=args.data_folder
    )

    trainer.train(model)

    if args.save_model:
        save_model(model, model.args, args.save_model)

    if args.generate_gif:
        train_loader, test_loader = data_processing.get_dataloader(
            data_folder=args.data_folder,
            batch_size=1,
            transform=lambda image: transformations.transformations_for_evaluation(image, crop_size=args.crop_size)
        )
        model.eval().to(device)
        batch = next(iter(test_loader)).to(device)
        generated_video = generator.generate_time_lapse_from_tensor(model, batch[:, :100], video_length=258)
        generated_video = transformations.unnormalize_image(generated_video)
        visualizer.visualize_tensor_images_as_gif(generated_video[0], path=args.generate_gif)


if __name__ == "__main__":
    train_model()