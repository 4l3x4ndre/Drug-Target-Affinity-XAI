import argparse
from xai.batch_analyzer import run_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch interaction analysis and save results.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument('--saved-model', type=str, default=None, help='Path to load saved model file.')
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to analyze.")
    parser.add_argument("--k-top-interactions", type=int, default=100, help="Top-k for Overlap Metric.")
    parser.add_argument("--top-x-for-correlation", type=int, default=100, help="Top-X for Pearson/Spearman Correlation.")
    parser.add_argument("--ig-steps", type=int, default=50, help="Number of steps for Integrated Gradients.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the analysis results.")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of samples to process in each batch.")
    args = parser.parse_args()

    run_analysis(device=args.device, 
                 saved_model=args.saved_model, 
                 num_samples=args.num_samples, 
                 k_top_interactions=args.k_top_interactions, 
                 top_x_for_correlation=args.top_x_for_correlation, 
                 ig_steps=args.ig_steps,
                 output_dir=args.output_dir,
                 batch_size=args.batch_size
                 )
