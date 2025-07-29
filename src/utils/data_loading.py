"""
Data loading utilities for extracting activations from language models
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import pickle

class ActivationDataset(Dataset):
    """Dataset for model activations"""
    
    def __init__(self, activations: torch.Tensor, metadata: Optional[Dict] = None):
        self.activations = activations
        self.metadata = metadata or {}
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return {'activations': self.activations[idx], 'index': idx}

def create_dataloader(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    layer_name: str,
    dataset_name: str = 'wikitext',
    split: str = 'train',
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    max_length: int = 128,
    device: str = 'cuda',
    cache_dir: Optional[Path] = None,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a dataloader for model activations
    
    Args:
        model: Language model to extract activations from
        tokenizer: Tokenizer for the model
        layer_name: Name of layer to extract activations from
        dataset_name: Name of dataset to use
        split: Dataset split (train/validation/test)
        batch_size: Batch size for activation extraction
        max_samples: Maximum number of samples to use
        max_length: Maximum sequence length
        device: Device to run on
        cache_dir: Directory to cache activations
        num_workers: Number of dataloader workers
        
    Returns:
        DataLoader yielding activation batches
    """
    # Check cache first
    if cache_dir is not None:
        cache_file = cache_dir / f"{model.__class__.__name__}_{layer_name}_{dataset_name}_{split}_{max_samples}.pkl"
        if cache_file.exists():
            print(f"Loading cached activations from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            dataset = ActivationDataset(cached_data['activations'], cached_data['metadata'])
            return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), 
                            num_workers=num_workers)
    
    # Load text dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        text_field = 'text'
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext', split=split)
        text_field = 'text'
    elif dataset_name == 'pile':
        dataset = load_dataset('EleutherAI/pile', split=split, streaming=True)
        text_field = 'text'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit samples if requested
    if max_samples is not None:
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        else:
            # For streaming datasets
            dataset = dataset.take(max_samples)
    
    # Extract activations
    print(f"Extracting activations from layer {layer_name}...")
    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=[sample[text_field] for sample in dataset],
        layer_name=layer_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device
    )
    
    # Cache activations if requested
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Caching activations to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'activations': activations,
                'metadata': {
                    'model': model.__class__.__name__,
                    'layer_name': layer_name,
                    'dataset': dataset_name,
                    'split': split,
                    'max_samples': max_samples
                }
            }, f)
    
    # Create dataset and dataloader
    activation_dataset = ActivationDataset(activations)