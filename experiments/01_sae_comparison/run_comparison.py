#!/usr/bin/env python3
"""
Comprehensive comparison of SAE architectures across multiple models
This experiment compares k-sparse, gated, vanilla, and our novel hybrid SAE
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
import time
from collections import defaultdict

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import KSparseSAE, HybridSAE
from src.models.vanilla_sae import VanillaSAE
from src.training.trainer import SAETrainer
#from src.analysis.metrics import compute_feature_metrics
from src.utils.model_loading import load_model
from src.utils.data_loading import create_dataloader
#from src.utils.hardware_utils import get_gpu_memory, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Compare SAE architectures')
    parser.add_argument('--model', type=str, default='gpt2', 
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'llama-7b', 'mistral-7b'])
    parser.add_argument('--layer', type=int, default=6,
                       help='Which layer to analyze')
    parser.add_argument('--n_features', type=int, default=32768,
                       help='Number of SAE features')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--use_wandb', action='store_true',
                       help='Log to Weights & Biases')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with less data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_sae_models(d_model: int, n_features: int, k_sparse: int, device: str):
    """Create all SAE model variants for comparison"""
    
    models = {
        'K-Sparse': KSparseSAE(
            d_model=d_model,
            n_features=n_features,
            k_sparse=k_sparse,
            device=device
        ),
        #'Gated': GatedSAE(
        #    d_model=d_model,
        #    n_features=n_features,
        #    device=device
        #),
        'Hybrid (Ours)': HybridSAE(
            d_model=d_model,
            n_features=n_features,
            k_sparse=k_sparse,
            approximation_features=n_features // 4,
            device=device
        ),
        'Vanilla': VanillaSAE(
            d_model=d_model,
            n_features=n_features,
            l1_coefficient=0.01,
            device=device
        )
    }
    
    return models

def evaluate_model(sae_model, dataloader, device):
    """Comprehensive evaluation of a single SAE model"""
    
    metrics = {
        'reconstruction_errors': [],
        'sparsity_levels': [],
        'active_features_per_batch': [],
        'inference_times': [],
        'memory_usage': []
    }
    
    # Track feature activations for dead feature analysis
    feature_activation_counts = torch.zeros(sae_model.n_features, device=device)
    total_samples = 0
    
    sae_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Extract data
            if isinstance(batch, dict):
                x = batch['activations'].to(device)
            else:
                x = batch.to(device)
            
            # Time inference
            start_time = time.time()
            output = sae_model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Extract outputs
            reconstruction = output['reconstruction']
            features = output['features']
            
            # Compute metrics
            mse = F.mse_loss(x, reconstruction, reduction='none').mean(dim=1)
            sparsity = (features > 0).float().sum(dim=1)
            active_features = (features > 0).any(dim=0).sum()
            
            # Store metrics
            metrics['reconstruction_errors'].extend(mse.cpu().numpy())
            metrics['sparsity_levels'].extend(sparsity.cpu().numpy())
            metrics['active_features_per_batch'].append(active_features.item())
            metrics['inference_times'].append(inference_time)
            metrics['memory_usage'].append(get_gpu_memory())
            
            # Update feature activation counts
            feature_activation_counts += (features > 0).float().sum(dim=0)
            total_samples += x.shape[0]
    
    # Compute aggregate statistics
    activation_rate = feature_activation_counts / total_samples
    
    final_metrics = {
        'reconstruction_error_mean': np.mean(metrics['reconstruction_errors']),
        'reconstruction_error_std': np.std(metrics['reconstruction_errors']),
        'sparsity_level_mean': np.mean(metrics['sparsity_levels']),
        'sparsity_level_std': np.std(metrics['sparsity_levels']),
        'active_features_mean': np.mean(metrics['active_features_per_batch']),
        'dead_features': (activation_rate < 0.001).sum().item(),
        'dead_feature_ratio': (activation_rate < 0.001).sum().item() / sae_model.n_features,
        'inference_time_mean': np.mean(metrics['inference_times']),
        'memory_usage_mean': np.mean(metrics['memory_usage']),
        'feature_activation_rate': activation_rate.cpu().numpy()
    }
    
    return final_metrics, metrics

def train_sae_model(model_name, sae_model, train_loader, val_loader, args, config, device):
    """Train a single SAE model"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = SAETrainer(
        sae_model=sae_model,
        learning_rate=args.learning_rate,
        weight_decay=config['training'].get('weight_decay', 1e-4),
        warmup_steps=config['training'].get('warmup_steps', 1000),
        device=device
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * args.epochs
    trainer.setup_scheduler(total_steps, args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_reconstruction_error': [],
        'val_sparsity_level': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)
        history['train_loss'].append(train_metrics['loss'])
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_reconstruction_error'].append(val_metrics['val_reconstruction_loss'])
        history['val_sparsity_level'].append(val_metrics['val_sparsity_level'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Reconstruction: {val_metrics['val_reconstruction_loss']:.4f}")
        print(f"  Val Sparsity: {val_metrics['val_sparsity_level']:.1f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                f"{model_name}/train_loss": train_metrics['loss'],
                f"{model_name}/val_loss": val_metrics['val_loss'],
                f"{model_name}/val_reconstruction": val_metrics['val_reconstruction_loss'],
                f"{model_name}/val_sparsity": val_metrics['val_sparsity_level'],
                f"{model_name}/learning_rate": train_metrics['learning_rate']
            })
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_epoch = epoch
        
        # Resample dead features periodically
        if epoch % 2 == 0 and epoch > 0:
            sample_data = next(iter(train_loader))
            if isinstance(sample_data, dict):
                sample_x = sample_data['activations'].to(device)
            else:
                sample_x = sample_data.to(device)
            
            n_resampled = sae_model.resample_dead_features(sample_x)
            print(f"  Resampled {n_resampled} dead features")
    
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    return history

def create_comparison_plots(all_results, save_dir):
    """Generate comprehensive comparison plots"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        history = results['training_history']
        color = colors[idx]
        
        # Training loss
        axes[0, 0].plot(history['train_loss'], label=model_name, 
                       color=color, linewidth=2)
        axes[0, 0].set_title('Training Loss', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Validation loss
        axes[0, 1].plot(history['val_loss'], label=model_name, 
                       color=color, linewidth=2)
        axes[0, 1].set_title('Validation Loss', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Reconstruction error
        axes[1, 0].plot(history['val_reconstruction_error'], label=model_name, 
                       color=color, linewidth=2)
        axes[1, 0].set_title('Validation Reconstruction Error', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        
        # Sparsity level
        axes[1, 1].plot(history['val_sparsity_level'], label=model_name, 
                       color=color, linewidth=2)
        axes[1, 1].set_title('Validation Sparsity Level', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Active Features')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison bar plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(all_results.keys())
    x_pos = np.arange(len(model_names))
    
    # Reconstruction error
    recon_errors = [results['eval_metrics']['reconstruction_error_mean'] 
                   for results in all_results.values()]
    recon_stds = [results['eval_metrics']['reconstruction_error_std'] 
                 for results in all_results.values()]
    
    axes[0, 0].bar(x_pos, recon_errors, yerr=recon_stds, capsize=5, color=colors)
    axes[0, 0].set_title('Reconstruction Error', fontsize=14)
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Sparsity level
    sparsity_means = [results['eval_metrics']['sparsity_level_mean'] 
                     for results in all_results.values()]
    sparsity_stds = [results['eval_metrics']['sparsity_level_std'] 
                    for results in all_results.values()]
    
    axes[0, 1].bar(x_pos, sparsity_means, yerr=sparsity_stds, capsize=5, color=colors)
    axes[0, 1].set_title('Sparsity Level', fontsize=14)
    axes[0, 1].set_ylabel('Active Features')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Dead features
    dead_ratios = [results['eval_metrics']['dead_feature_ratio'] * 100 
                  for results in all_results.values()]
    
    axes[1, 0].bar(x_pos, dead_ratios, color=colors)
    axes[1, 0].set_title('Dead Features Percentage', fontsize=14)
    axes[1, 0].set_ylabel('Dead Features (%)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Inference time
    inference_times = [results['eval_metrics']['inference_time_mean'] 
                      for results in all_results.values()]
    
    axes[1, 1].bar(x_pos, inference_times, color=colors)
    axes[1, 1].set_title('Inference Time', fontsize=14)
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency trade-off scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        recon_error = results['eval_metrics']['reconstruction_error_mean']
        inference_time = results['eval_metrics']['inference_time_mean']
        memory = results['eval_metrics']['memory_usage_mean']
        
        # Size represents memory usage
        ax.scatter(inference_time, recon_error, 
                  s=memory/10, label=model_name,
                  color=colors[idx], alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('Inference Time (ms)', fontsize=12)
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_title('Performance vs Efficiency Trade-off', fontsize=14)
    ax.legend()
    
    # Add annotation
    ax.text(0.95, 0.95, 'Bubble size = Memory Usage', 
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature activation heatmap (for Hybrid model)
    if 'Hybrid (Ours)' in all_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        activation_rates = all_results['Hybrid (Ours)']['eval_metrics']['feature_activation_rate']
        
        # Sort and take top 1000 features
        sorted_indices = np.argsort(activation_rates)[::-1][:1000]
        top_features = activation_rates[sorted_indices]
        
        # Reshape for visualization
        grid_size = int(np.sqrt(len(top_features)))
        heatmap_data = top_features[:grid_size**2].reshape(grid_size, grid_size)
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_title('Hybrid Model - Top Feature Activation Rates', fontsize=14)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_dir / 'hybrid_feature_activation.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_table(all_results):
    """Create a summary comparison table"""
    
    summary_data = []
    
    for model_name, results in all_results.items():
        metrics = results['eval_metrics']
        
        summary_data.append({
            'Model': model_name,
            'Reconstruction Error': f"{metrics['reconstruction_error_mean']:.4f} ± {metrics['reconstruction_error_std']:.4f}",
            'Sparsity Level': f"{metrics['sparsity_level_mean']:.1f} ± {metrics['sparsity_level_std']:.1f}",
            'Active Features': f"{metrics['active_features_mean']:.0f}",
            'Dead Features (%)': f"{metrics['dead_feature_ratio']*100:.1f}%",
            'Inference Time (ms)': f"{metrics['inference_time_mean']:.2f}",
            'Memory (MB)': f"{metrics['memory_usage_mean']:.1f}"
        })
    
    df = pd.DataFrame(summary_data)
    return df

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"sae_comparison_{args.model}_layer{args.layer}_{timestamp}"
    exp_dir = Path('results') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="mechanistic-interpretability-sae",
            name=exp_name,
            config={**vars(args), **config}
        )
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load language model
    print(f"\nLoading {args.model}...")
    lm_model, tokenizer = load_model(args.model)
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    # Get model dimension
    if hasattr(lm_model.config, 'hidden_size'):
        d_model = lm_model.config.hidden_size
    elif hasattr(lm_model.config, 'd_model'):
        d_model = lm_model.config.d_model
    else:
        raise ValueError("Cannot determine model dimension")
    
    print(f"Model dimension: {d_model}")
    
    # Determine layer name format
    if 'gpt2' in args.model:
        layer_name = f"transformer.h.{args.layer}"
    elif 'llama' in args.model:
        layer_name = f"model.layers.{args.layer}"
    elif 'mistral' in args.model:
        layer_name = f"model.layers.{args.layer}"
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    
    if args.debug:
        max_train_samples = 1000
        max_val_samples = 200
    else:
        max_train_samples = config.get('max_train_samples', 100000)
        max_val_samples = config.get('max_val_samples', 10000)
    
    train_loader = create_dataloader(
        model=lm_model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        dataset_name=config.get('dataset', 'wikitext'),
        batch_size=args.batch_size,
        max_samples=max_train_samples,
        device=device,
        cache_dir=Path('data/activations')
    )
    
    val_loader = create_dataloader(
        model=lm_model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        dataset_name=config.get('dataset', 'wikitext'),
        batch_size=args.batch_size,
        max_samples=max_val_samples,
        split='validation',
        device=device,
        cache_dir=Path('data/activations')
    )
    
    # Create SAE models
    print(f"\nCreating SAE models...")
    k_sparse = config.get('k_sparse', 128)
    sae_models = create_sae_models(d_model, args.n_features, k_sparse, device)
    
    # Results storage
    all_results = {}
    
    # Train and evaluate each model
    for model_name, sae_model in sae_models.items():
        # Train model
        history = train_sae_model(
            model_name=model_name,
            sae_model=sae_model,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            config=config,
            device=device
        )
        
        # Evaluate model
        print(f"\nEvaluating {model_name}...")
        eval_metrics, raw_metrics = evaluate_model(sae_model, val_loader, device)
        
        # Store results
        all_results[model_name] = {
            'training_history': history,
            'eval_metrics': eval_metrics,
            'raw_metrics': raw_metrics,
            'model_config': {
                'd_model': d_model,
                'n_features': args.n_features,
                'k_sparse': k_sparse
            }
        }
        
        # Save model checkpoint
        checkpoint_path = exp_dir / f"{model_name.replace(' ', '_')}_checkpoint.pt"
        torch.save({
            'model_state_dict': sae_model.state_dict(),
            'config': all_results[model_name]['model_config'],
            'eval_metrics': eval_metrics
        }, checkpoint_path)
        
        # Print evaluation results
        print(f"\nResults for {model_name}:")
        print(f"  Reconstruction Error: {eval_metrics['reconstruction_error_mean']:.4f} ± {eval_metrics['reconstruction_error_std']:.4f}")
        print(f"  Sparsity Level: {eval_metrics['sparsity_level_mean']:.1f} ± {eval_metrics['sparsity_level_std']:.1f}")
        print(f"  Dead Features: {eval_metrics['dead_features']} ({eval_metrics['dead_feature_ratio']:.1%})")
        print(f"  Inference Time: {eval_metrics['inference_time_mean']:.2f}ms")
        print(f"  Memory Usage: {eval_metrics['memory_usage_mean']:.2f}MB")
    
    # Special analysis for Hybrid model
    if 'Hybrid (Ours)' in sae_models:
        print(f"\nAnalyzing Hybrid model routing efficiency...")
        routing_stats = sae_models['Hybrid (Ours)'].analyze_routing_efficiency(val_loader)
        all_results['Hybrid (Ours)']['routing_stats'] = routing_stats
        
        print(f"Routing Statistics:")
        print(f"  Fast path usage: {routing_stats['routing_ratio_fast']:.1%}")
        print(f"  Deep path usage: {routing_stats['routing_ratio_deep']:.1%}")
        print(f"  Complexity threshold: {routing_stats['threshold']:.3f}")
        if 'reconstruction_error_fast' in routing_stats:
            print(f"  Fast path error: {routing_stats['reconstruction_error_fast']:.4f}")
        if 'reconstruction_error_deep' in routing_stats:
            print(f"  Deep path error: {routing_stats['reconstruction_error_deep']:.4f}")
    
    # Generate comparison plots
    print(f"\nGenerating comparison plots...")
    create_comparison_plots(all_results, exp_dir)
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    summary_df.to_csv(exp_dir / 'summary_results.csv', index=False)
    print(f"\nSummary Results:")
    print(summary_df.to_string())
    
    # Save detailed results
    results_to_save = {}
    for model_name, results in all_results.items():
        results_to_save[model_name] = {
            'training_history': results['training_history'],
            'eval_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in results['eval_metrics'].items() 
                           if k != 'feature_activation_rate'},
            'model_config': results['model_config']
        }
        if 'routing_stats' in results:
            results_to_save[model_name]['routing_stats'] = results['routing_stats']
    
    with open(exp_dir / 'detailed_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Log to wandb
    if args.use_wandb:
        # Log summary table
        wandb.log({"summary_table": wandb.Table(dataframe=summary_df)})
        
        # Log plots
        for plot_file in exp_dir.glob('*.png'):
            wandb.log({plot_file.stem: wandb.Image(str(plot_file))})
        
        # Log best metrics
        for model_name, results in all_results.items():
            metrics = results['eval_metrics']
            wandb.summary[f"{model_name}/best_reconstruction_error"] = metrics['reconstruction_error_mean']
            wandb.summary[f"{model_name}/best_sparsity_level"] = metrics['sparsity_level_mean']
            wandb.summary[f"{model_name}/dead_feature_ratio"] = metrics['dead_feature_ratio']
        
        wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()