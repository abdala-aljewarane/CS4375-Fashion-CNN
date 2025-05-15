import os
import subprocess
import argparse
import pandas as pd
import sys
from datetime import datetime
import time

def print_gpu_info():
    """Print GPU information if available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU will be used for training.\n")
        else:
            print("CUDA available: No")
            print("Training will use CPU only.\n")
    except Exception as e:
        print(f"Error checking GPU info: {e}")
        print("Training will use CPU only.\n")

def run_experiment(dataset, model_name, epochs, transfer_learning=False, source_dataset=None):
    """Run a single experiment"""
    print("="*80)
    print(f"EXPERIMENT: {model_name} on {dataset}")
    print("="*80)
    
    # Set up experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{dataset}_{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join("code", "train_simple.py"),
        f"--dataset={dataset}",
        f"--model={model_name}",
        f"--epochs={epochs}",
        f"--scheduler=cosine",
        f"--results-dir={exp_dir}"
    ]
    
    # If using transfer learning, add flag
    if transfer_learning and source_dataset:
        # Determine the path to the source model
        source_exp_dirs = [d for d in os.listdir("experiments") if d.startswith(f"{source_dataset}_{model_name}")]
        if source_exp_dirs:
            # Use the most recent experiment
            source_exp_dir = sorted(source_exp_dirs)[-1]
            source_model_path = os.path.join("experiments", source_exp_dir, "best_model.pth")
            if os.path.exists(source_model_path):
                cmd.append(f"--pretrained-model={source_model_path}")
                print(f"Using pretrained model from {source_model_path}")
    
    # Run the command
    subprocess.run(cmd)
    
    return exp_dir

def main(args):
    """Run experiments with multiple models on multiple datasets"""
    results = []
    
    # Keep track of how long each experiment takes
    for dataset in args.datasets:
        for model in args.models:
            start_time = time.time()
            
            # Check if we should use transfer learning
            transfer_learning = (dataset == 'deep_fashion' and 'fashion_mnist' in args.datasets)
            source_dataset = 'fashion_mnist' if transfer_learning else None
            
            exp_dir = run_experiment(dataset, model, args.epochs, 
                                    transfer_learning=transfer_learning, 
                                    source_dataset=source_dataset)
            
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            
            results.append({
                'dataset': dataset,
                'model': model,
                'training_time_minutes': duration_minutes
            })
            
    # Print summary
    print("\nEXPERIMENT SUMMARY:")
    print("=" * 19)
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    # Check GPU availability
    print_gpu_info()
    
    parser = argparse.ArgumentParser(description='Run CNN experiments on multiple datasets')
    parser.add_argument('--datasets', nargs='+', default=['fashion_mnist', 'cifar10'],
                        help='datasets to use (default: fashion_mnist cifar10)')
    parser.add_argument('--models', nargs='+', default=['lenet', 'vgg16', 'custom'],
                        help='models to use (default: lenet vgg16 custom)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs for each experiment (default: 25)')
    parser.add_argument('--transfer-learning', action='store_true',
                        help='use transfer learning from fashion_mnist to deep_fashion')
    
    args = parser.parse_args()

    main(args) 