#!/usr/bin/env python3
"""
Simple runner script for Google Colab
Just upload this file and run: python colab_simple_run.py
"""

import os
import sys
import subprocess

def setup_environment():
    """Install required packages"""
    print("Installing required packages...")
    packages = [
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
        "einops"
    ]
    
    for package in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("✓ Packages installed!")

def check_gpu():
    """Check GPU availability"""
    import torch
    print(f"\nGPU Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return torch.cuda.is_available()

def run_experiment(model="gpt2", n_features=8192, batch_size=16, epochs=5):
    """Run the SAE comparison experiment"""
    
    # Create command
    cmd = [
        "python", "experiments/01_sae_comparison/run_comparison.py",
        "--model", model,
        "--layer", "6",
        "--n_features", str(n_features),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--seed", "42"
    ]
    
    print(f"\nRunning experiment with:")
    print(f"  Model: {model}")
    print(f"  Features: {n_features}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print("\nThis will take 1-3 hours...\n")
    
    # Run experiment
    subprocess.run(cmd)

def main():
    """Main runner"""
    print("="*60)
    print("SAE Comparison Experiment - Colab Runner")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("\n⚠️  No GPU detected! Go to Runtime > Change runtime type > GPU")
        return
    
    # Determine settings based on GPU memory
    import torch
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory > 20:  # A100 or better
        print("\n🚀 High-end GPU detected - using full settings")
        n_features = 16384
        batch_size = 32
        epochs = 10
    elif gpu_memory > 15:  # V100 or T4
        print("\n✓ Standard GPU detected - using optimized settings")
        n_features = 8192
        batch_size = 16
        epochs = 5
    else:  # Older GPUs
        print("\n⚠️  Limited GPU memory - using reduced settings")
        n_features = 4096
        batch_size = 8
        epochs = 3
    
    # Run experiment
    run_experiment(
        model="gpt2",
        n_features=n_features,
        batch_size=batch_size,
        epochs=epochs
    )
    
    print("\n✓ Experiment complete! Check the 'results' folder for outputs.")

if __name__ == "__main__":
    main()