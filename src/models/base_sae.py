"""
Abstract base class for Sparse Autoencoders
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List

class BaseSAE(nn.Module, ABC):
    """Abstract base class for Sparse Autoencoders"""
    
    def __init__(self, 
                 d_model: int,
                 n_features: int,
                 device: str = 'cuda'):
        """
        Initialize base SAE
        
        Args:
            d_model: Dimension of model activations
            n_features: Number of sparse features
            device: Device to run on (cuda/cpu)
        """
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.device = device
        
        # Initialize weights with Xavier initialization
        self.W_enc = nn.Parameter(torch.randn(n_features, d_model, device=device) / np.sqrt(d_model))
        self.W_dec = nn.Parameter(torch.randn(d_model, n_features, device=device) / np.sqrt(n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features, device=device))
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device))
        
        # Normalize decoder weights
        with torch.no_grad():
            self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=0)
        
        self.to(device)
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features"""
        pass
    
    @abstractmethod
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode features back to activations"""
        pass
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning reconstruction and features
        
        Args:
            x: Input activations [batch_size, d_model]
            
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed activations
                - features: Sparse feature activations
                - loss: Total loss (reconstruction + sparsity)
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        
        return {
            'reconstruction': reconstruction,
            'features': features,
            'loss': self.compute_loss(x, reconstruction, features)
        }
    
    def compute_loss(self, 
                     x: torch.Tensor, 
                     x_hat: torch.Tensor, 
                     f: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reconstruction + sparsity loss"""
        reconstruction_loss = torch.nn.functional.mse_loss(x, x_hat)
        sparsity_loss = self.sparsity_penalty(f)
        
        total_loss = reconstruction_loss + sparsity_loss
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'sparsity': sparsity_loss
        }
    
    @abstractmethod
    def sparsity_penalty(self, f: torch.Tensor) -> torch.Tensor:
        """Compute sparsity penalty"""
        pass
    
    def get_feature_activations(self, 
                                model: nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                layer_name: str,
                                max_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract feature activations for a dataset
        
        Args:
            model: Language model to analyze
            dataloader: DataLoader for input data
            layer_name: Name of layer to extract from
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary with feature statistics
        """
        activations = []
        
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            activations.append(output.detach())
        
        # Register hook
        hooks = []
        for name, module in model.named_modules():
            if name == layer_name:
                hooks.append(module.register_forward_hook(hook_fn))
                break
        
        if not hooks:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Collect activations
        samples_processed = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and samples_processed >= max_samples:
                    break
                    
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                _ = model(batch)
                samples_processed += len(batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process collected activations
        all_activations = torch.cat(activations, dim=0)
        if len(all_activations.shape) == 3:  # [batch, seq_len, hidden]
            all_activations = all_activations.reshape(-1, all_activations.shape[-1])
        
        # Encode all activations
        all_features = []
        batch_size = 1024
        
        for i in range(0, len(all_activations), batch_size):
            batch = all_activations[i:i+batch_size]
            features = self.encode(batch)
            all_features.append(features)
        
        all_features = torch.cat(all_features, dim=0)
        
        # Compute statistics
        feature_counts = (all_features > 0).float().sum(dim=0)
        feature_means = all_features.mean(dim=0)
        feature_stds = all_features.std(dim=0)
        
        return {
            'features': all_features,
            'counts': feature_counts,
            'means': feature_means,
            'stds': feature_stds,
            'activation_rate': feature_counts / len(all_features),
            'dead_features': (feature_counts == 0).sum().item()
        }
    
    def resample_dead_features(self, 
                              activation_data: torch.Tensor,
                              threshold: float = 0.01) -> int:
        """
        Resample dead features with new random directions
        
        Args:
            activation_data: Sample of activation data
            threshold: Minimum activation rate to not be considered dead
            
        Returns:
            Number of resampled features
        """
        with torch.no_grad():
            # Get feature activation statistics
            features = self.encode(activation_data)
            activation_rate = (features > 0).float().mean(dim=0)
            
            # Identify dead features
            dead_mask = activation_rate < threshold
            n_dead = dead_mask.sum().item()
            
            if n_dead > 0:
                # Compute PCA of activation data
                U, S, V = torch.svd(activation_data.T)
                
                # Reinitialize dead features with top PCA components
                # (with some noise to break symmetry)
                for i, is_dead in enumerate(dead_mask):
                    if is_dead:
                        # Pick a random PCA component
                        component_idx = torch.randint(0, min(50, V.shape[1]), (1,)).item()
                        new_direction = V[:, component_idx] + 0.1 * torch.randn_like(V[:, component_idx])
                        
                        # Update encoder weight
                        self.W_enc.data[i] = new_direction / new_direction.norm()
                        
                        # Reset bias
                        self.b_enc.data[i] = 0
        
        return n_dead
    
    def get_feature_importance(self, 
                              features: torch.Tensor,
                              target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute importance scores for features
        
        Args:
            features: Feature activations
            target: Optional target for supervised importance
            
        Returns:
            Importance scores for each feature
        """
        # Basic importance: activation frequency * magnitude
        activation_freq = (features > 0).float().mean(dim=0)
        mean_magnitude = features.abs().mean(dim=0)
        
        importance = activation_freq * mean_magnitude
        
        # If target provided, weight by correlation with target
        if target is not None:
            correlations = []
            for i in range(features.shape[1]):
                if features[:, i].std() > 0:
                    corr = torch.corrcoef(torch.stack([features[:, i], target]))[0, 1]
                    correlations.append(corr.abs())
                else:
                    correlations.append(torch.tensor(0.0))
            
            correlations = torch.stack(correlations)
            importance = importance * correlations
        
        return importance