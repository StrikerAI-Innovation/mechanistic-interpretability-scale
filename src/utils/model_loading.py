"""
Utilities for loading pre-trained language models
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    MistralForCausalLM,
    AutoConfig
)
from typing import Tuple, Optional
import os
from pathlib import Path

# Model registry
MODEL_REGISTRY = {
    'gpt2': ('gpt2', GPT2Model, GPT2Tokenizer),
    'gpt2-medium': ('gpt2-medium', GPT2Model, GPT2Tokenizer),
    'gpt2-large': ('gpt2-large', GPT2Model, GPT2Tokenizer),
    'gpt2-xl': ('gpt2-xl', GPT2Model, GPT2Tokenizer),
    'llama-7b': ('meta-llama/Llama-2-7b-hf', LlamaForCausalLM, LlamaTokenizer),
    'llama-13b': ('meta-llama/Llama-2-13b-hf', LlamaForCausalLM, LlamaTokenizer),
    'mistral-7b': ('mistralai/Mistral-7B-v0.1', MistralForCausalLM, AutoTokenizer),
}

def load_model(
    model_name: str,
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    load_in_8bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a pre-trained language model and tokenizer
    
    Args:
        model_name: Name of the model to load
        device: Device to load model on
        cache_dir: Directory to cache downloaded models
        load_in_8bit: Whether to load in 8-bit precision
        torch_dtype: Data type for model weights
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_id, model_class, tokenizer_class = MODEL_REGISTRY[model_name]
    
    # Set default cache directory
    if cache_dir is None:
        cache_dir = Path.home() / '.cache' / 'huggingface'
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set dtype
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print(f"Loading {model_name} from {model_id}...")
    
    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    load_kwargs = {
        'cache_dir': cache_dir,
        'torch_dtype': torch_dtype,
        'trust_remote_code': True,
    }
    
    if load_in_8bit and device == 'cuda':
        load_kwargs['load_in_8bit'] = True
        load_kwargs['device_map'] = 'auto'
    else:
        load_kwargs['device_map'] = device
    
    # For causal LM models
    if model_class in [LlamaForCausalLM, MistralForCausalLM]:
        model = model_class.from_pretrained(model_id, **load_kwargs)
    else:
        # For GPT2 models
        model = model_class.from_pretrained(model_id, **load_kwargs)
        if device != 'cpu' and not load_in_8bit:
            model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} with {total_params / 1e9:.2f}B parameters")
    
    return model, tokenizer

def get_model_info(model_name: str) -> dict:
    """
    Get information about a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_id, _, _ = MODEL_REGISTRY[model_name]
    config = AutoConfig.from_pretrained(model_id)
    
    info = {
        'model_id': model_id,
        'model_name': model_name,
        'hidden_size': getattr(config, 'hidden_size', getattr(config, 'd_model', None)),
        'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', None)),
        'num_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_heads', None)),
        'vocab_size': config.vocab_size,
        'max_position_embeddings': getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', None)),
    }
    
    return info

def get_activation_hook(layer_name: str):
    """
    Create a hook function for extracting activations
    
    Args:
        layer_name: Name of the layer to hook
        
    Returns:
        Hook function
    """
    activations = []
    
    def hook_fn(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        
        # Detach and store
        activations.append(activation.detach())
    
    return hook_fn, activations