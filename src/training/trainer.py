"""
SAE Trainer implementation with optimization strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Optional, Dict, Tuple, List
import numpy as np
from tqdm import tqdm

class SAETrainer:
    """
    Trainer for Sparse Autoencoders with advanced optimization strategies
    """
    
    def __init__(self,
                 sae_model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 warmup_steps: int = 1000,
                 max_grad_norm: float = 1.0,
                 scheduler_type: str = 'cosine',
                 device: str = 'cuda'):
        """
        Initialize SAE trainer
        
        Args:
            sae_model: SAE model to train
            learning_rate: Peak learning rate
            weight_decay: L2 regularization
            warmup_steps: Linear warmup steps
            max_grad_norm: Gradient clipping threshold
            scheduler_type: 'cosine', 'onecycle', or 'none'
            device: Device to train on
        """
        self.model = sae_model
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Track training statistics
        self.loss_history = []
        self.grad_norm_history = []
        
    def train_step(self, x: torch.Tensor) -> Tuple[float, float]:
        """
        Single training step
        
        Args:
            x: Input activations
            
        Returns:
            loss: Training loss
            grad_norm: Gradient norm
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x)
        
        # Get total loss
        if isinstance(output['loss'], dict):
            loss = output['loss']['total']
        else:
            loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduling
        self._update_lr()
        
        # Update step counter
        self.current_step += 1
        
        # Track statistics
        self.loss_history.append(loss.item())
        self.grad_norm_history.append(grad_norm.item())
        
        return loss.item(), grad_norm.item()
    
    def _update_lr(self):
        """Update learning rate based on schedule"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale
        elif self.scheduler_type == 'cosine' and hasattr(self, 'scheduler'):
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   epoch: int,
                   log_interval: int = 100) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data
            epoch: Current epoch number
            log_interval: Steps between logging
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        epoch_loss = 0
        epoch_reconstruction_loss = 0
        epoch_sparsity_loss = 0
        epoch_grad_norm = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract activations
            if isinstance(batch, dict):
                x = batch['activations'].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Training step
            loss, grad_norm = self.train_step(x)
            
            # Get detailed losses if available
            with torch.no_grad():
                output = self.model(x)
                if 'losses' in output:
                    reconstruction_loss = output['losses']['reconstruction'].item()
                    sparsity_loss = output['losses']['sparsity'].item()
                else:
                    reconstruction_loss = loss
                    sparsity_loss = 0
            
            # Accumulate metrics
            epoch_loss += loss
            epoch_reconstruction_loss += reconstruction_loss
            epoch_sparsity_loss += sparsity_loss
            epoch_grad_norm += grad_norm
            num_batches += 1
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{self.get_lr():.2e}",
                    'grad': f"{grad_norm:.2f}"
                })
        
        # Compute epoch averages
        metrics = {
            'loss': epoch_loss / num_batches,
            'reconstruction_loss': epoch_reconstruction_loss / num_batches,
            'sparsity_loss': epoch_sparsity_loss / num_batches,
            'grad_norm': epoch_grad_norm / num_batches,
            'learning_rate': self.get_lr()
        }
        
        return metrics
    
    def validate(self, 
                dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model performance
        
        Args:
            dataloader: Validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_reconstruction_loss = 0
        total_sparsity_loss = 0
        total_sparsity_level = 0
        total_active_features = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Extract activations
                if isinstance(batch, dict):
                    x = batch['activations'].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Forward pass
                output = self.model(x)
                
                # Get losses
                if isinstance(output['loss'], dict):
                    loss = output['loss']['total']
                    reconstruction_loss = output['loss']['reconstruction']
                    sparsity_loss = output['loss']['sparsity']
                else:
                    loss = output['loss']
                    reconstruction_loss = loss
                    sparsity_loss = torch.tensor(0)
                
                # Get sparsity metrics
                features = output['features']
                sparsity_level = (features > 0).float().sum(dim=-1).mean()
                active_features = (features > 0).any(dim=0).sum()
                
                # Accumulate
                total_loss += loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_sparsity_loss += sparsity_loss.item()
                total_sparsity_level += sparsity_level.item()
                total_active_features += active_features.item()
                num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_reconstruction_loss': total_reconstruction_loss / num_batches,
            'val_sparsity_loss': total_sparsity_loss / num_batches,
            'val_sparsity_level': total_sparsity_level / num_batches,
            'val_active_features': total_active_features / num_batches
        }
        
        return metrics
    
    def setup_scheduler(self, 
                       total_steps: int,
                       epochs: int = None):
        """
        Setup learning rate scheduler
        
        Args:
            total_steps: Total training steps
            epochs: Number of epochs (for cosine scheduler)
        """
        if self.scheduler_type == 'cosine' and epochs is not None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif self.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
    
    def save_checkpoint(self, 
                       path: str,
                       epoch: int,
                       metrics: Dict[str, float]):
        """
        Save training checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'loss_history': self.loss_history,
            'grad_norm_history': self.grad_norm_history,
            'current_step': self.current_step
        }
        
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load training checkpoint
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.loss_history = checkpoint.get('loss_history', [])
        self.grad_norm_history = checkpoint.get('grad_norm_history', [])
        self.current_step = checkpoint.get('current_step', 0)
        
        return checkpoint