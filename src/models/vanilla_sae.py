"""
Vanilla Sparse Autoencoder implementation with L1 regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
from .base_sae import BaseSAE

class VanillaSAE(BaseSAE):
    """
    Standard Sparse Autoencoder with L1 sparsity penalty
    This serves as a baseline for comparison
    """
    
    def __init__(self,
                 d_model: int,
                 n_features: int,
                 l1_coefficient: float = 0.01,
                 use_bias_decoder: bool = True,
                 tied_weights: bool = False,
                 device: str = 'cuda'):
        """
        Initialize Vanilla SAE
        
        Args:
            d_model: Model dimension
            n_features: Number of features
            l1_coefficient: L1 penalty strength
            use_bias_decoder: Whether to use bias in decoder
            tied_weights: Whether to tie encoder/decoder weights
            device: Device to run on
        """
        super().__init__(d_model, n_features, device)
        
        self.l1_coefficient = l1_coefficient
        self.use_bias_decoder = use_bias_decoder
        self.tied_weights = tied_weights
        
        if not use_bias_decoder:
            # Remove decoder bias
            self.b_dec = nn.Parameter(torch.zeros(d_model, device=device), requires_grad=False)
        
        if tied_weights:
            # Tie weights (decoder = encoder.T)
            self.W_dec = None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations with ReLU nonlinearity
        
        Args:
            x: Input activations [batch_size, d_model]
            
        Returns:
            Sparse features [batch_size, n_features]
        """
        pre_activation = x @ self.W_enc.T + self.b_enc
        return F.relu(pre_activation)
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activations
        
        Args:
            f: Features [batch_size, n_features]
            
        Returns:
            Reconstructed activations [batch_size, d_model]
        """
        if self.tied_weights:
            # Use transposed encoder weights
            reconstruction = f @ self.W_enc + self.b_dec
        else:
            reconstruction = f @ self.W_dec.T + self.b_dec
        
        return reconstruction
    
    def sparsity_penalty(self, f: torch.Tensor) -> torch.Tensor:
        """
        L1 sparsity penalty
        
        Args:
            f: Feature activations
            
        Returns:
            Sparsity penalty loss
        """
        return self.l1_coefficient * f.abs().mean()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with detailed metrics
        
        Args:
            x: Input activations
            
        Returns:
            Dictionary with reconstruction and metrics
        """
        # Encode
        features = self.encode(x)
        
        # Decode
        reconstruction = self.decode(features)
        
        # Compute losses
        losses = self.compute_loss(x, reconstruction, features)
        
        # Compute additional metrics
        sparsity_level = (features > 0).float().sum(dim=-1).mean()
        active_features = (features > 0).any(dim=0).sum()
        
        return {
            'reconstruction': reconstruction,
            'features': features,
            'loss': losses['total'],
            'losses': losses,
            'sparsity_level': sparsity_level,
            'active_features': active_features,
            'l0_norm': (features > 0).float().sum(dim=-1).mean(),
            'l1_norm': features.abs().sum(dim=-1).mean()
        }