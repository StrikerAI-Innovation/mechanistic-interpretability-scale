"""
K-Sparse Autoencoder implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from .base_sae import BaseSAE

class KSparseSAE(BaseSAE):
    """
    K-Sparse Autoencoder with top-k activation selection
    This is currently the gold standard for avoiding dead features
    """
    
    def __init__(self,
                 d_model: int,
                 n_features: int,
                 k_sparse: int = 128,
                 use_batch_norm: bool = False,
                 device: str = 'cuda'):
        """
        Initialize K-Sparse SAE
        
        Args:
            d_model: Model dimension
            n_features: Number of features
            k_sparse: Number of active features per sample
            use_batch_norm: Whether to use batch normalization
            device: Device to run on
        """
        super().__init__(d_model, n_features, device)
        
        self.k_sparse = k_sparse
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn_enc = nn.BatchNorm1d(n_features)
            self.bn_dec = nn.BatchNorm1d(d_model)
        
        # Temperature parameter for soft top-k
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode with top-k sparsity constraint
        
        Args:
            x: Input activations [batch_size, d_model]
            
        Returns:
            Sparse features [batch_size, n_features]
        """
        # Linear transformation
        pre_activation = x @ self.W_enc.T + self.b_enc
        
        # Apply batch norm if enabled
        if self.use_batch_norm and self.training:
            pre_activation = self.bn_enc(pre_activation)
        
        # Apply ReLU
        activated = F.relu(pre_activation)
        
        # Apply top-k sparsity
        if self.k_sparse < self.n_features:
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(activated, self.k_sparse, dim=-1)
            
            # Create sparse tensor
            sparse_features = torch.zeros_like(activated)
            sparse_features.scatter_(dim=-1, index=topk_indices, src=topk_values)
            
            return sparse_features
        else:
            # If k >= n_features, no sparsity constraint
            return activated
    
    def encode_soft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode with soft top-k for differentiable sparsity
        Used during training for better gradients
        """
        x = x.float()
        pre_activation = x @ self.W_enc.T + self.b_enc
        
        if self.use_batch_norm and self.training:
            pre_activation = self.bn_enc(pre_activation)
        
        activated = F.relu(pre_activation)
        
        # Soft top-k using temperature
        if self.k_sparse < self.n_features:
            # Compute soft mask
            topk_values, _ = torch.topk(activated, self.k_sparse, dim=-1)
            threshold = topk_values[..., -1:].detach()
            
            # Soft threshold using sigmoid
            mask = torch.sigmoid((activated - threshold) * self.temperature)
            
            return activated * mask
        else:
            return activated
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activations
        
        Args:
            f: Sparse features [batch_size, n_features]
            
        Returns:
            Reconstructed activations [batch_size, d_model]
        """
        reconstruction = f @ self.W_dec.T + self.b_dec
        
        if self.use_batch_norm and self.training:
            reconstruction = self.bn_dec(reconstruction)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor, use_soft: bool = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional soft encoding
        
        Args:
            x: Input activations
            use_soft: Whether to use soft top-k (defaults to self.training)
            
        Returns:
            Dictionary with reconstruction, features, and losses
        """
        if use_soft is None:
            use_soft = self.training
        
        # Encode
        if use_soft:
            features = self.encode_soft(x)
        else:
            features = self.encode(x)
        
        # Decode
        reconstruction = self.decode(features)
        
        # Compute losses
        losses = self.compute_loss(x, reconstruction, features)
        
        return {
            'reconstruction': reconstruction,
            'features': features,
            'loss': losses['total'],
            'losses': losses,
            'sparsity_level': (features > 0).float().sum(dim=-1).mean(),
            'active_features': (features > 0).any(dim=0).sum()
        }
    
    def sparsity_penalty(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity penalty for k-sparse features
        
        Since we enforce top-k directly, we use a mild L1 penalty
        on the active features to encourage smaller magnitudes
        """
        # Only penalize active features
        active_mask = (f > 0).float()
        l1_penalty = (f * active_mask).sum(dim=-1).mean()
        
        # Add penalty for deviation from target sparsity
        actual_sparsity = active_mask.sum(dim=-1).mean()
        target_sparsity = self.k_sparse
        sparsity_deviation = (actual_sparsity - target_sparsity).abs()
        
        return 0.01 * l1_penalty + 0.001 * sparsity_deviation
    
    def get_dead_feature_mask(self, 
                             dataloader: torch.utils.data.DataLoader,
                             threshold: float = 0.001) -> torch.Tensor:
        """
        Identify dead features across a dataset
        
        Args:
            dataloader: DataLoader to analyze
            threshold: Minimum activation rate
            
        Returns:
            Boolean mask of dead features
        """
        total_activations = torch.zeros(self.n_features, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['activations']
                else:
                    x = batch
                
                x = x.to(self.device)
                features = self.encode(x)
                
                total_activations += (features > 0).float().sum(dim=0)
                total_samples += x.shape[0]
        
        activation_rate = total_activations / total_samples
        return activation_rate < threshold
    
    def compute_feature_similarity(self) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between encoder weights
        
        Returns:
            Similarity matrix [n_features, n_features]
        """
        # Normalize encoder weights
        W_normalized = F.normalize(self.W_enc, dim=1)
        
        # Compute pairwise similarities
        similarity = W_normalized @ W_normalized.T
        
        return similarity
    
    def prune_redundant_features(self, similarity_threshold: float = 0.95) -> int:
        """
        Identify and prune highly similar features
        
        Args:
            similarity_threshold: Threshold for considering features redundant
            
        Returns:
            Number of pruned features
        """
        similarity = self.compute_feature_similarity()
        
        # Mask diagonal
        similarity.fill_diagonal_(0)
        
        # Find redundant pairs
        redundant_pairs = (similarity > similarity_threshold).nonzero()
        
        # Keep track of features to prune
        to_prune = set()
        
        for i, j in redundant_pairs:
            i, j = i.item(), j.item()
            # Only prune if neither already marked
            if i not in to_prune and j not in to_prune:
                # Prune the one with lower average activation
                if self.W_enc[i].abs().mean() < self.W_enc[j].abs().mean():
                    to_prune.add(i)
                else:
                    to_prune.add(j)
        
        # Zero out pruned features
        with torch.no_grad():
            for idx in to_prune:
                self.W_enc.data[idx] = 0
                self.W_dec.data[:, idx] = 0
                self.b_enc.data[idx] = -1e10  # Effectively disable
        
        return len(to_prune)