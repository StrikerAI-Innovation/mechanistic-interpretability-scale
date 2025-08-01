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

def extract_activations(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_name: str,
    max_length: int = 128,
    batch_size: int = 32,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract activations from a specific layer of the model
    
    Args:
        model: Language model
        tokenizer: Tokenizer for the model
        texts: List of text strings
        layer_name: Name of layer to extract from (e.g., 'transformer.h.6')
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        device: Device to run on
        
    Returns:
        Tensor of activations [num_samples, d_model]
    """
    model.eval()
    model = model.to(device)
    
    # Find the target layer
    target_layer = get_layer_by_name(model, layer_name)
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Storage for activations
    all_activations = []
    
    # Hook to capture activations
    def hook_fn(module, input, output):
        # Handle different output formats
        if isinstance(output, tuple):
            activation = output[0]  # Usually the first element is the hidden states
        else:
            activation = output
        
        # If activation has sequence dimension, pool over it
        if len(activation.shape) == 3:  # [batch, seq_len, hidden_dim]
            # Use mean pooling over sequence length
            activation = activation.mean(dim=1)  # [batch, hidden_dim]
        
        all_activations.append(activation.detach().cpu())
    
    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Process texts in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
                batch_texts = texts[i:i + batch_size]
                
                # Filter out empty texts
                batch_texts = [text for text in batch_texts if text and text.strip()]
                if not batch_texts:
                    continue
                
                # Tokenize batch
                try:
                    inputs = tokenizer(
                        batch_texts,
                        max_length=max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Forward pass (this will trigger the hook)
                    _ = model(**inputs)
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size}: {e}")
                    continue
    
    finally:
        # Remove hook
        handle.remove()
    
    if not all_activations:
        raise ValueError("No activations were extracted")
    
    # Concatenate all activations
    activations = torch.cat(all_activations, dim=0)
    
    print(f"Extracted {activations.shape[0]} activations of dimension {activations.shape[1]}")
    
    return activations

def get_layer_by_name(model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
    """
    Get a layer from the model by its name with improved error handling
    
    Args:
        model: The model
        layer_name: Dot-separated layer name (e.g., 'transformer.h.6')
        
    Returns:
        The layer module or None if not found
    """
    parts = layer_name.split('.')
    current = model
    
    print(f"Trying to access layer: {layer_name}")
    print(f"Parts: {parts}")
    
    for i, part in enumerate(parts):
        print(f"  Step {i}: Looking for '{part}' in {type(current).__name__}")
        
        if hasattr(current, part):
            current = getattr(current, part)
            print(f"    ✓ Found: {type(current).__name__}")
        else:
            # Print available attributes for debugging
            available = [name for name, _ in current.named_children()]
            print(f"    ✗ Not found. Available children: {available}")
            return None
    
    print(f"Successfully found layer: {type(current).__name__}")
    return current

def get_all_layer_names(model: torch.nn.Module, max_depth: int = 3) -> List[str]:
    """
    Get all possible layer names in the model up to a certain depth.
    """
    def _get_names(module, prefix="", depth=0):
        names = []
        if depth >= max_depth:
            return names
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            names.append(full_name)
            names.extend(_get_names(child, full_name, depth + 1))
        
        return names
    
    return _get_names(model)

def find_gpt2_layers(model) -> List[str]:
    """
    Find GPT-2 transformer block layers specifically.
    """
    layer_names = []
    
    # GPT-2 typically has structure: transformer.h.{i} where i is layer index
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
        print(f"Found {num_layers} transformer layers")
        
        for i in range(num_layers):
            layer_names.append(f"transformer.h.{i}")
    
    return layer_names

def auto_find_layer(model, target_layer_idx: int = 6) -> Optional[str]:
    """
    Automatically find the correct layer name for a given layer index.
    """
    print(f"Auto-finding layer {target_layer_idx}...")
    
    # Try common patterns
    patterns = [
        f"h.{target_layer_idx}",
        f"transformer.h.{target_layer_idx}",
        f"layers.{target_layer_idx}",
        f"transformer.layers.{target_layer_idx}",
        f"model.layers.{target_layer_idx}",
        f"transformer.block.{target_layer_idx}",
    ]
    
    for pattern in patterns:
        if get_layer_by_name(model, pattern) is not None:
            print(f"Found working pattern: {pattern}")
            return pattern
    
    # If patterns don't work, search through all layers
    all_names = get_all_layer_names(model)
    
    # Look for layer names containing the target index
    candidates = [name for name in all_names if str(target_layer_idx) in name]
    print(f"Candidate layers containing '{target_layer_idx}': {candidates}")
    
    # Try each candidate
    for candidate in candidates:
        layer = get_layer_by_name(model, candidate)
        if layer is not None and hasattr(layer, 'forward'):
            print(f"Found working candidate: {candidate}")
            return candidate
    
    return None

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
    
    # Extract texts from dataset
    print(f"Extracting text from {len(dataset)} samples...")
    texts = []
    for sample in tqdm(dataset, desc="Loading texts"):
        text = sample[text_field]
        # Filter out very short texts
        if text and len(text.strip()) > 10:
            texts.append(text.strip())
    
    # Limit texts if max_samples specified
    if max_samples is not None:
        texts = texts[:max_samples]
    
    print(f"Processing {len(texts)} texts...")
    
    # Extract activations
    print(f"Extracting activations from layer {layer_name}...")
    activations = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_name=layer_name,
        max_length=max_length,
        batch_size=32,  # Use smaller batch for activation extraction
        device=device
    )
    
    # Cache activations if requested
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{model.__class__.__name__}_{layer_name}_{dataset_name}_{split}_{max_samples}.pkl"
        print(f"Caching activations to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'activations': activations,
                'metadata': {
                    'model': model.__class__.__name__,
                    'layer_name': layer_name,
                    'dataset': dataset_name,
                    'split': split,
                    'max_samples': max_samples,
                    'num_samples': len(activations),
                    'd_model': activations.shape[1]
                }
            }, f)
    
    # Create dataset and dataloader
    activation_dataset = ActivationDataset(activations, {
        'layer_name': layer_name,
        'dataset_name': dataset_name,
        'split': split
    })
    
    return DataLoader(
        activation_dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

def load_cached_activations(cache_file: Path) -> Optional[torch.Tensor]:
    """Load cached activations if available"""
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['activations']
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None
    return None

def get_activation_stats(activations: torch.Tensor) -> Dict[str, float]:
    """Get statistics about activations"""
    return {
        'mean': activations.mean().item(),
        'std': activations.std().item(),
        'min': activations.min().item(),
        'max': activations.max().item(),
        'sparsity': (activations == 0).float().mean().item(),
        'shape': list(activations.shape)
    }

def print_model_structure(model, max_depth=3, current_depth=0, prefix=""):
    """
    Print the structure of a model to find correct layer names.
    """
    if current_depth > max_depth:
        return
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print("  " * current_depth + f"{full_name}: {type(module).__name__}")
        
        # If it's a transformer layer, print its children too
        if hasattr(module, 'named_children') and current_depth < max_depth:
            print_model_structure(module, max_depth, current_depth + 1, full_name)

def find_transformer_layers(model):
    """
    Find all transformer layer names in the model.
    """
    layer_names = []
    
    def find_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this looks like a transformer layer
            if ('layer' in name.lower() or 'block' in name.lower() or 
                'h' in name or 'transformer' in name.lower()):
                layer_names.append(full_name)
            
            # Recursively search children
            find_layers(child, full_name)
    
    find_layers(model)
    return layer_names

# Quick test function you can run
def debug_gpt2_layers():
    from transformers import AutoModel
    
    print("Loading GPT-2 model for inspection...")
    model = AutoModel.from_pretrained("gpt2")
    
    print("\n" + "="*50)
    print("MODEL STRUCTURE:")
    print("="*50)
    print_model_structure(model, max_depth=2)
    
    print("\n" + "="*50)
    print("POTENTIAL TRANSFORMER LAYERS:")
    print("="*50)
    layers = find_transformer_layers(model)
    for i, layer in enumerate(layers):
        print(f"{i}: {layer}")
    
    print("\n" + "="*50)
    print("TESTING LAYER ACCESS:")
    print("="*50)
    
    # Test different naming conventions
    test_names = [
        "transformer.h.6",
        "h.6", 
        "transformer.h.6.mlp",
        "transformer.h.6.attn",
        "layers.6",
        "transformer.layers.6"
    ]
    
    for name in test_names:
        try:
            layer = get_layer_by_name(model, name)
            if layer is not None:
                print(f"✓ Found: {name} -> {type(layer).__name__}")
            else:
                print(f"✗ Not found: {name}")
        except Exception as e:
            print(f"✗ Error accessing {name}: {e}")
    
    return model