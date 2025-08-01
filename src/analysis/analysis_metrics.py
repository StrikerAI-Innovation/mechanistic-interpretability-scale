"""
Analysis metrics for evaluating SAE model performance and interpretability.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureMetrics:
    """Container for feature-level metrics."""
    activation_frequency: float
    selectivity_score: float
    interpretability_score: float
    polysemanticity_score: float
    dead_feature: bool

@dataclass
class ModelMetrics:
    """Container for model-level metrics."""
    reconstruction_error: float
    sparsity_level: float
    feature_utilization: float
    dead_feature_ratio: float
    l0_norm: float
    l1_norm: float
    explained_variance: float

def compute_feature_metrics(
    features: torch.Tensor,
    activations: torch.Tensor,
    feature_labels: Optional[List[str]] = None
) -> Dict[int, FeatureMetrics]:
    """
    Compute comprehensive metrics for individual features.
    
    Args:
        features: Feature activations [batch_size, n_features]
        activations: Original activations [batch_size, d_model]
        feature_labels: Optional human-readable labels for features
        
    Returns:
        Dictionary mapping feature indices to FeatureMetrics
    """
    n_features = features.shape[1]
    batch_size = features.shape[0]
    
    metrics = {}
    
    for feature_idx in range(n_features):
        feature_acts = features[:, feature_idx]
        
        # Activation frequency (how often this feature fires)
        activation_frequency = (feature_acts > 0).float().mean().item()
        
        # Selectivity score (how selective is this feature)
        selectivity_score = compute_selectivity(feature_acts)
        
        # Interpretability score (placeholder - would need human evaluation)
        interpretability_score = compute_interpretability_proxy(feature_acts, activations)
        
        # Polysemanticity score (does this feature represent multiple concepts)
        polysemanticity_score = compute_polysemanticity(feature_acts)
        
        # Dead feature check
        dead_feature = activation_frequency < 1e-5
        
        metrics[feature_idx] = FeatureMetrics(
            activation_frequency=activation_frequency,
            selectivity_score=selectivity_score,
            interpretability_score=interpretability_score,
            polysemanticity_score=polysemanticity_score,
            dead_feature=dead_feature
        )
    
    return metrics

def compute_selectivity(feature_activations: torch.Tensor) -> float:
    """
    Compute selectivity score for a feature.
    Higher scores indicate more selective (sparse) activation patterns.
    """
    if feature_activations.sum() == 0:
        return 0.0
    
    # Normalize activations
    normalized = feature_activations / (feature_activations.max() + 1e-8)
    
    # Compute selectivity using normalized entropy
    # More selective features have lower entropy
    hist, _ = torch.histogram(normalized[normalized > 0], bins=50)
    hist = hist.float() + 1e-8  # Avoid log(0)
    prob = hist / hist.sum()
    
    ent = -torch.sum(prob * torch.log(prob)).item()
    max_entropy = np.log(50)  # Maximum entropy for 50 bins
    
    selectivity = 1.0 - (ent / max_entropy)  # Higher = more selective
    return max(0.0, selectivity)

def compute_interpretability_proxy(feature_acts: torch.Tensor, original_acts: torch.Tensor) -> float:
    """
    Compute a proxy for interpretability based on feature consistency.
    This is a placeholder - true interpretability requires human evaluation.
    """
    if feature_acts.sum() == 0:
        return 0.0
    
    # Simple proxy: correlation with principal components
    try:
        # When feature is active, how consistent is the original activation pattern?
        active_mask = feature_acts > 0
        if active_mask.sum() < 2:
            return 0.0
        
        active_original = original_acts[active_mask]
        
        # Compute variance explained by first principal component
        centered = active_original - active_original.mean(dim=0)
        U, S, V = torch.svd(centered)
        
        total_var = torch.sum(S ** 2)
        explained_var = S[0] ** 2 / total_var if total_var > 0 else 0.0
        
        return explained_var.item()
    
    except Exception as e:
        logger.warning(f"Failed to compute interpretability proxy: {e}")
        return 0.0

def compute_polysemanticity(feature_acts: torch.Tensor) -> float:
    """
    Compute polysemanticity score - whether feature represents multiple concepts.
    Lower scores indicate more monosemantic (single concept) features.
    """
    if feature_acts.sum() == 0:
        return 1.0  # Dead features are maximally polysemantic
    
    # Simple heuristic: features with multi-modal activation distributions
    # are more likely to be polysemantic
    active_acts = feature_acts[feature_acts > 0]
    
    if len(active_acts) < 10:
        return 0.5  # Not enough data
    
    # Compute histogram and look for multiple modes
    hist, bin_edges = torch.histogram(active_acts, bins=min(20, len(active_acts) // 2))
    
    # Find peaks in histogram
    peaks = find_peaks(hist.numpy())
    
    # More peaks suggest more concepts (polysemanticity)
    polysemanticity = min(1.0, len(peaks) / 3.0)  # Normalize by expected max peaks
    
    return polysemanticity

def find_peaks(histogram: np.ndarray, min_height: float = 0.1) -> List[int]:
    """Find peaks in a histogram."""
    peaks = []
    
    for i in range(1, len(histogram) - 1):
        if (histogram[i] > histogram[i-1] and 
            histogram[i] > histogram[i+1] and 
            histogram[i] > min_height * histogram.max()):
            peaks.append(i)
    
    return peaks

def compute_model_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    features: torch.Tensor
) -> ModelMetrics:
    """
    Compute comprehensive model-level metrics.
    
    Args:
        original: Original activations [batch_size, d_model]
        reconstructed: Reconstructed activations [batch_size, d_model]
        features: Feature activations [batch_size, n_features]
        
    Returns:
        ModelMetrics object
    """
    batch_size, d_model = original.shape
    n_features = features.shape[1]
    
    # Reconstruction error
    mse = F.mse_loss(reconstructed, original)
    reconstruction_error = mse.item()
    
    # Sparsity metrics
    l0_norm = (features > 0).float().sum(dim=1).mean().item()
    l1_norm = features.abs().sum(dim=1).mean().item()
    sparsity_level = l0_norm  # Average number of active features
    
    # Feature utilization
    active_features = (features > 0).any(dim=0).sum().item()
    feature_utilization = active_features / n_features
    
    # Dead feature ratio
    dead_features = ((features > 0).sum(dim=0) == 0).sum().item()
    dead_feature_ratio = dead_features / n_features
    
    # Explained variance
    original_var = original.var().item()
    residual_var = (original - reconstructed).var().item()
    explained_variance = 1.0 - (residual_var / original_var) if original_var > 0 else 0.0
    
    return ModelMetrics(
        reconstruction_error=reconstruction_error,
        sparsity_level=sparsity_level,
        feature_utilization=feature_utilization,
        dead_feature_ratio=dead_feature_ratio,
        l0_norm=l0_norm,
        l1_norm=l1_norm,
        explained_variance=explained_variance
    )

def compute_feature_clustering_metrics(features: torch.Tensor) -> Dict[str, float]:
    """
    Compute clustering metrics to assess feature organization.
    
    Args:
        features: Feature activations [batch_size, n_features]
        
    Returns:
        Dictionary of clustering metrics
    """
    # Only consider samples where at least one feature is active
    active_samples = features.sum(dim=1) > 0
    
    if active_samples.sum() < 10:
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': float('inf'),
            'feature_correlation_mean': 0.0,
            'feature_correlation_std': 0.0
        }
    
    active_features = features[active_samples]
    
    try:
        # Simple clustering based on which features are active
        labels = []
        for sample in active_features:
            # Create label based on most active features
            top_features = torch.topk(sample, k=min(5, (sample > 0).sum().item()))[1]
            label = hash(tuple(sorted(top_features.tolist()))) % 100  # Simple hash-based clustering
            labels.append(label)
        
        labels = np.array(labels)
        
        # Only compute if we have multiple clusters
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(active_features.numpy(), labels)
            davies_bouldin = davies_bouldin_score(active_features.numpy(), labels)
        else:
            silhouette = 0.0
            davies_bouldin = float('inf')
    
    except Exception as e:
        logger.warning(f"Failed to compute clustering metrics: {e}")
        silhouette = 0.0
        davies_bouldin = float('inf')
    
    # Feature correlation analysis
    feature_corr = torch.corrcoef(features.T)
    # Remove diagonal and NaN values
    mask = ~torch.eye(feature_corr.shape[0], dtype=torch.bool)
    correlations = feature_corr[mask]
    correlations = correlations[~torch.isnan(correlations)]
    
    if len(correlations) > 0:
        corr_mean = correlations.mean().item()
        corr_std = correlations.std().item()
    else:
        corr_mean = 0.0
        corr_std = 0.0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'feature_correlation_mean': corr_mean,
        'feature_correlation_std': corr_std
    }

def compute_information_theoretic_metrics(
    original: torch.Tensor,
    features: torch.Tensor
) -> Dict[str, float]:
    """
    Compute information-theoretic metrics.
    
    Args:
        original: Original activations [batch_size, d_model]
        features: Feature activations [batch_size, n_features]
        
    Returns:
        Dictionary of information metrics
    """
    metrics = {}
    
    # Mutual information approximation
    try:
        # Discretize activations for entropy computation
        original_discrete = discretize_tensor(original)
        features_discrete = discretize_tensor(features)
        
        # Compute joint and marginal entropies
        joint_entropy = compute_joint_entropy(original_discrete, features_discrete)
        original_entropy = compute_entropy(original_discrete)
        features_entropy = compute_entropy(features_discrete)
        
        # Mutual information
        mutual_info = original_entropy + features_entropy - joint_entropy
        
        # Normalized mutual information
        normalized_mi = mutual_info / max(original_entropy, features_entropy) if max(original_entropy, features_entropy) > 0 else 0.0
        
        metrics['mutual_information'] = mutual_info
        metrics['normalized_mutual_information'] = normalized_mi
        metrics['original_entropy'] = original_entropy
        metrics['features_entropy'] = features_entropy
        
    except Exception as e:
        logger.warning(f"Failed to compute information metrics: {e}")
        metrics.update({
            'mutual_information': 0.0,
            'normalized_mutual_information': 0.0,
            'original_entropy': 0.0,
            'features_entropy': 0.0
        })
    
    return metrics

def discretize_tensor(tensor: torch.Tensor, n_bins: int = 50) -> torch.Tensor:
    """Discretize tensor values for entropy computation."""
    # Flatten and remove outliers
    flat = tensor.flatten()
    q01, q99 = torch.quantile(flat, torch.tensor([0.01, 0.99]))
    clipped = torch.clamp(flat, q01, q99)
    
    # Create bins
    bins = torch.linspace(clipped.min(), clipped.max(), n_bins + 1)
    discretized = torch.bucketize(clipped, bins[1:-1])
    
    return discretized.reshape(tensor.shape)

def compute_entropy(discrete_tensor: torch.Tensor) -> float:
    """Compute entropy of discretized tensor."""
    unique, counts = torch.unique(discrete_tensor, return_counts=True)
    probs = counts.float() / counts.sum()
    ent = entropy(probs.numpy())
    return float(ent)

def compute_joint_entropy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute joint entropy of two discretized tensors."""
    # Flatten tensors
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    # Create joint representation
    max_val1 = flat1.max().item() + 1
    joint = flat1 * max_val1 + flat2
    
    unique, counts = torch.unique(joint, return_counts=True)
    probs = counts.float() / counts.sum()
    ent = entropy(probs.numpy())
    return float(ent)

def evaluate_feature_interpretability(
    features: torch.Tensor,
    activations: torch.Tensor,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Evaluate interpretability of top-k most active features.
    
    Args:
        features: Feature activations [batch_size, n_features]
        activations: Original activations [batch_size, d_model]
        top_k: Number of top features to analyze
        
    Returns:
        Dictionary with interpretability analysis
    """
    # Find most active features
    feature_activity = features.sum(dim=0)
    top_features = torch.topk(feature_activity, k=min(top_k, len(feature_activity)))[1]
    
    results = {
        'top_features': top_features.tolist(),
        'feature_metrics': {},
        'summary_stats': {}
    }
    
    # Compute metrics for top features
    feature_metrics = compute_feature_metrics(features, activations)
    
    selectivity_scores = []
    interpretability_scores = []
    polysemanticity_scores = []
    
    for feature_idx in top_features:
        idx = feature_idx.item()
        metrics = feature_metrics[idx]
        
        results['feature_metrics'][idx] = {
            'activation_frequency': metrics.activation_frequency,
            'selectivity_score': metrics.selectivity_score,
            'interpretability_score': metrics.interpretability_score,
            'polysemanticity_score': metrics.polysemanticity_score,
            'dead_feature': metrics.dead_feature
        }
        
        selectivity_scores.append(metrics.selectivity_score)
        interpretability_scores.append(metrics.interpretability_score)
        polysemanticity_scores.append(metrics.polysemanticity_score)
    
    # Summary statistics
    results['summary_stats'] = {
        'mean_selectivity': np.mean(selectivity_scores),
        'std_selectivity': np.std(selectivity_scores),
        'mean_interpretability': np.mean(interpretability_scores),
        'std_interpretability': np.std(interpretability_scores),
        'mean_polysemanticity': np.mean(polysemanticity_scores),
        'std_polysemanticity': np.std(polysemanticity_scores),
        'num_high_quality_features': sum(1 for s in selectivity_scores if s > 0.7),
        'num_interpretable_features': sum(1 for s in interpretability_scores if s > 0.5)
    }
    
    return results

def compare_model_quality(metrics_list: List[ModelMetrics], model_names: List[str]) -> Dict[str, Any]:
    """
    Compare multiple models across quality metrics.
    
    Args:
        metrics_list: List of ModelMetrics objects
        model_names: List of model names
        
    Returns:
        Comparison results
    """
    comparison = {
        'models': model_names,
        'metrics': {},
        'rankings': {},
        'best_model': {}
    }
    
    # Extract metrics
    metric_names = ['reconstruction_error', 'sparsity_level', 'feature_utilization', 
                   'dead_feature_ratio', 'explained_variance']
    
    for metric_name in metric_names:
        values = [getattr(metrics, metric_name) for metrics in metrics_list]
        comparison['metrics'][metric_name] = values
        
        # Rank models (lower is better for errors, higher for positive metrics)
        if metric_name in ['reconstruction_error', 'dead_feature_ratio']:
            rankings = np.argsort(values)  # Lower is better
        else:
            rankings = np.argsort(values)[::-1]  # Higher is better
        
        comparison['rankings'][metric_name] = [model_names[i] for i in rankings]
    
    # Determine overall best model (simple scoring)
    scores = np.zeros(len(model_names))
    for metric_name in metric_names:
        rankings = comparison['rankings'][metric_name]
        for i, model in enumerate(model_names):
            rank = rankings.index(model)
            scores[i] += len(model_names) - rank  # Higher score for better rank
    
    best_idx = np.argmax(scores)
    comparison['best_model'] = {
        'name': model_names[best_idx],
        'score': scores[best_idx],
        'all_scores': scores.tolist()
    }
    
    return comparison