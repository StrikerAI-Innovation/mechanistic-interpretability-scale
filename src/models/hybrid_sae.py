"""
Novel Hybrid Sparse Autoencoder implementation
Dynamically routes between fast and deep analysis based on input complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from .base_sae import BaseSAE
from .k_sparse_sae import KSparseSAE


class ComplexityRouter(nn.Module):
    """Neural network to estimate input complexity"""
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute complexity scores and routing decisions
        
        Returns:
            complexity_scores: Continuous complexity scores [0, 1]
            use_deep: Boolean mask for deep analysis
        """
        x = x.to(next(self.parameters()).device)
        complexity_scores = self.network(x).squeeze(-1)
        use_deep = complexity_scores > self.threshold
        
        return complexity_scores, use_deep

class HybridSAE(BaseSAE):
    """
    Novel Hybrid Sparse Autoencoder
    
    Key innovations:
    1. Dynamic routing based on input complexity
    2. Shared decoder for consistency
    3. Adaptive sparsity based on feature utility
    4. Efficient approximation for real-time analysis
    """
    
    def __init__(self,
                 d_model: int,
                 n_features: int,
                 k_sparse: int = 128,
                 approximation_features: int = 8192,
                 router_hidden_dim: int = 256,
                 feature_dropout: float = 0.1,
                 device: str = 'cuda'):
        """
        Initialize Hybrid SAE
        
        Args:
            d_model: Model dimension
            n_features: Number of features for deep analysis
            k_sparse: Sparsity level
            approximation_features: Features for fast approximation
            router_hidden_dim: Hidden dimension for router network
            feature_dropout: Dropout rate for features
            device: Device to run on
        """
        super().__init__(d_model, n_features, device)
        
        self.k_sparse = k_sparse
        self.approximation_features = approximation_features
        
        # Fast approximator (smaller, faster)
        self.fast_encoder = nn.Sequential(
            nn.Linear(d_model, approximation_features),
            nn.ReLU(),
            nn.Dropout(feature_dropout)
        )
        
        # Deep analyzer (larger, more accurate)
        self.deep_encoder = nn.Sequential(
            nn.Linear(d_model, n_features * 2),
            nn.ReLU(),
            nn.LayerNorm(n_features * 2),
            nn.Dropout(feature_dropout),
            nn.Linear(n_features * 2, n_features),
            nn.ReLU()
        )
        
        # Shared decoder for consistency
        self.shared_decoder = nn.Sequential(
            nn.Linear(n_features, d_model * 2),
            nn.ReLU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(feature_dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Complexity router
        self.router = ComplexityRouter(d_model, router_hidden_dim)
        
        # Feature utility estimator
        self.utility_estimator = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_features),
            nn.Sigmoid()
        )
        
        # Mapping from fast to full feature space
        self.feature_projector = nn.Linear(approximation_features, n_features)
        
        # Track routing statistics
        self.register_buffer('routing_stats', torch.zeros(2))  # [fast_count, deep_count]
        
        # Now move entire model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode(self, x: torch.Tensor, force_mode: Optional[str] = None) -> torch.Tensor:
        """
        Adaptively encode based on input complexity
        
        Args:
            x: Input activations [batch_size, d_model]
            force_mode: Force 'fast' or 'deep' mode (for analysis)
            
        Returns:
            Sparse features [batch_size, n_features]
        """
        # Get the model's device from one of its parameters
        device = next(self.parameters()).device
        x = x.to(device)
        batch_size = x.shape[0]
        
        # Get routing decisions
        if force_mode is None:
            complexity_scores, use_deep = self.router(x)
        elif force_mode == 'fast':
            use_deep = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        else:  # force_mode == 'deep'
            use_deep = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        
        # Initialize output
        features = torch.zeros(batch_size, self.n_features, device=x.device)
        
        # Route to appropriate encoder
        if use_deep.any():
            deep_indices = use_deep.nonzero().squeeze(-1)
            deep_features = self.deep_encoder(x[deep_indices])
            
            # Apply k-sparse constraint
            deep_features = self._apply_k_sparse(deep_features)
            features[deep_indices] = deep_features
            
            # Update statistics
            self.routing_stats[1] += deep_indices.numel()
        
        if (~use_deep).any():
            fast_indices = (~use_deep).nonzero().squeeze(-1)
            fast_features = self.fast_encoder(x[fast_indices])
            
            # Apply k-sparse constraint
            fast_features = self._apply_k_sparse(fast_features, k=self.k_sparse // 2)
            
            # Project to full feature space
            projected_features = self.feature_projector(fast_features)
            projected_features = F.relu(projected_features)
            
            # Apply sparsity again after projection
            projected_features = self._apply_k_sparse(projected_features)
            features[fast_indices] = projected_features
            
            # Update statistics
            self.routing_stats[0] += fast_indices.numel()
        
        return features
    
    def _apply_k_sparse(self, features: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """Apply k-sparse constraint to features"""
        if k is None:
            k = self.k_sparse
        
        if k < features.shape[-1]:
            topk_values, topk_indices = torch.topk(features, k, dim=-1)
            sparse_features = torch.zeros_like(features)
            sparse_features.scatter_(dim=-1, index=topk_indices, src=topk_values)
            return sparse_features
        else:
            return features
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activations using shared decoder
        
        Args:
            f: Sparse features [batch_size, n_features]
            
        Returns:
            Reconstructed activations [batch_size, d_model]
        """
        return self.shared_decoder(f)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with routing and adaptive sparsity
        
        Args:
            x: Input activations [batch_size, d_model]
            
        Returns:
            Dictionary with outputs and metrics
        """
        x = x.to(next(self.parameters()).device)
        # Encode with routing
        features = self.encode(x)
        
        # Apply adaptive sparsity based on utility
        utility_weights = self.utility_estimator(features.detach())
        features_weighted = features * utility_weights
        
        # Decode
        reconstruction = self.decode(features_weighted)
        
        # Compute losses
        losses = self.compute_loss(x, reconstruction, features)
        
        # Get routing statistics
        complexity_scores, use_deep = self.router(x)
        
        return {
            'reconstruction': reconstruction,
            'features': features,
            'features_weighted': features_weighted,
            'loss': losses['total'],
            'losses': losses,
            'complexity_scores': complexity_scores,
            'routing_ratio': use_deep.float().mean(),
            'sparsity_level': (features > 0).float().sum(dim=-1).mean(),
            'utility_weights': utility_weights.mean(dim=0)
        }
    
    def sparsity_penalty(self, f: torch.Tensor) -> torch.Tensor:
        """
        Adaptive sparsity penalty based on feature utility
        """
        # Basic L1 penalty
        l1_penalty = f.abs().mean()
        
        # Adaptive penalty based on activation patterns
        activation_rate = (f > 0).float().mean(dim=0)
        
        # Penalize features that are always on or always off
        utility = 4 * activation_rate * (1 - activation_rate)  # Maximum at 0.5
        adaptive_penalty = (f.abs() * (1 - utility)).mean()
        
        return 0.1 * l1_penalty + 0.05 * adaptive_penalty
    
    def analyze_routing_efficiency(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Analyze routing decisions and efficiency
        
        Args:
            dataloader: Data to analyze
            
        Returns:
            Dictionary with routing statistics
        """
        total_fast = 0
        total_deep = 0
        complexity_scores_all = []
        reconstruction_errors_fast = []
        reconstruction_errors_deep = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['activations']
                else:
                    x = batch
                
                x = x.to(self.device)
                
                # Get routing decisions
                complexity_scores, use_deep = self.router(x)
                complexity_scores_all.append(complexity_scores)
                
                # Analyze reconstruction quality for each mode
                if use_deep.any():
                    deep_x = x[use_deep]
                    deep_features = self.encode(deep_x, force_mode='deep')
                    deep_recon = self.decode(deep_features)
                    deep_error = F.mse_loss(deep_x, deep_recon, reduction='none').mean(dim=1)
                    reconstruction_errors_deep.append(deep_error)
                    total_deep += use_deep.sum().item()
                
                if (~use_deep).any():
                    fast_x = x[~use_deep]
                    fast_features = self.encode(fast_x, force_mode='fast')
                    fast_recon = self.decode(fast_features)
                    fast_error = F.mse_loss(fast_x, fast_recon, reduction='none').mean(dim=1)
                    reconstruction_errors_fast.append(fast_error)
                    total_fast += (~use_deep).sum().item()
        
        # Aggregate statistics
        total_samples = total_fast + total_deep
        complexity_scores_all = torch.cat(complexity_scores_all)
        
        stats = {
            'routing_ratio_fast': total_fast / total_samples,
            'routing_ratio_deep': total_deep / total_samples,
            'complexity_mean': complexity_scores_all.mean().item(),
            'complexity_std': complexity_scores_all.std().item(),
            'threshold': self.router.threshold.item()
        }
        
        if reconstruction_errors_fast:
            stats['reconstruction_error_fast'] = torch.cat(reconstruction_errors_fast).mean().item()
        
        if reconstruction_errors_deep:
            stats['reconstruction_error_deep'] = torch.cat(reconstruction_errors_deep).mean().item()
        
        return stats
    
    def get_feature_importance_adaptive(self, 
                                       dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Compute feature importance with routing-aware weighting
        
        Args:
            dataloader: Data to analyze
            
        Returns:
            Feature importance scores [n_features]
        """
        importance_scores = torch.zeros(self.n_features, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['activations']
                else:
                    x = batch
                
                x = x.to(self.device)
                
                # Get features and utility weights
                output = self.forward(x)
                features = output['features']
                utility_weights = output['utility_weights']
                
                # Weight importance by both activation and utility
                batch_importance = (features > 0).float().mean(dim=0) * utility_weights
                importance_scores += batch_importance * x.shape[0]
                total_samples += x.shape[0]
        
        return importance_scores / total_samples