import argparse
from .rule_based_generator import RuleBasedGenerator, STD_DEVIATIONS
import src.utils.utils as utils
import src.visualizer.visualizer as visualizer


def main():
    parser = argparse.ArgumentParser(description='Rule-based video generator for cell tracking simulations')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV/GZ file with cell tracks data')
    
    # Optional arguments
    parser.add_argument('--site', type=int, default=1,
                       help='Image metadata site to use (default: 1)')
    parser.add_argument('--exp-id', type=int, default=1,
                       help='Experiment ID to use (default: 1)')
    parser.add_argument('--frames', type=int, default=258,
                       help='Number of frames to generate (default: 258)')
    parser.add_argument('--output', type=str,
                       help='Path to save output CSV (optional)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of the simulation')
    
    parser.add_argument('--mutation-type', type=str, default='WT',
                    choices=list(STD_DEVIATIONS.keys()),
                    help='Mutation type to simulate (default: WT)')

    args = parser.parse_args()

    # Load and prepare data
    df = utils.unpack_and_read(args.input)
    df_first_frame = df[(df['Image_Metadata_Site'] == args.site) & 
                       (df['Exp_ID'] == args.exp_id) & 
                       (df['Image_Metadata_T'] == 0)][
        ['track_id', 'objNuclei_Location_Center_X', 
         'objNuclei_Location_Center_Y', 'ERKKTR_ratio', 'Image_Metadata_T']]
    
    # Generate video
    generator = RuleBasedGenerator(df_first_frame=df_first_frame, 
                                 number_of_frames=args.frames,
                                 mutation_type=args.mutation_type
                                 )
    video_data = generator.generate_time_lapse()
    
    # Save output if requested
    if args.output:
        video_data.to_csv(args.output, index=False)
        print(f"Saved generated video to {args.output}")
    
    # Visualize if requested
    if args.visualize:
        visualizer.visualize_simulation(video_data)


if __name__ == "__main__":
    main()