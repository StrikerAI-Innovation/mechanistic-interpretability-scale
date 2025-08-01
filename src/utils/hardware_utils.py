"""
Hardware utilities for monitoring GPU memory, setting seeds, and performance optimization.
"""

import torch
import numpy as np
import random
import psutil
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")

def get_gpu_memory() -> float:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        # Get memory usage for current device
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert to MB
        return memory_allocated
    except Exception as e:
        logger.warning(f"Failed to get GPU memory: {e}")
        return 0.0

def get_gpu_memory_info() -> Dict[str, float]:
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'cached': 0.0,
            'total': 0.0,
            'free': 0.0,
            'utilization': 0.0
        }
    
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
        cached = torch.cuda.memory_reserved(device) / 1024 / 1024  # MB
        total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024  # MB
        free = total - allocated
        utilization = (allocated / total) * 100 if total > 0 else 0.0
        
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'free': free,
            'utilization': utilization
        }
    except Exception as e:
        logger.warning(f"Failed to get detailed GPU memory info: {e}")
        return {
            'allocated': 0.0,
            'cached': 0.0,
            'total': 0.0,
            'free': 0.0,
            'utilization': 0.0
        }

def get_cpu_memory_info() -> Dict[str, float]:
    """Get CPU memory information."""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024 / 1024,  # MB
            'available': memory.available / 1024 / 1024,  # MB
            'used': memory.used / 1024 / 1024,  # MB
            'percentage': memory.percent
        }
    except Exception as e:
        logger.warning(f"Failed to get CPU memory info: {e}")
        return {
            'total': 0.0,
            'available': 0.0,
            'used': 0.0,
            'percentage': 0.0
        }

def optimize_torch_settings():
    """Optimize PyTorch settings for performance."""
    # Enable optimized kernels
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set number of threads for CPU operations
    num_cores = psutil.cpu_count(logical=False)
    torch.set_num_threads(num_cores)
    
    logger.info(f"Torch optimization enabled. CPU threads: {num_cores}")

def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_device_info() -> Dict[str, any]:
    """Get comprehensive device information."""
    info = {
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'cpu_memory': get_cpu_memory_info(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['cuda_devices'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'device_id': i,
                'name': props.name,
                'total_memory': props.total_memory / 1024 / 1024,  # MB
                'multiprocessor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor
            }
            info['cuda_devices'].append(device_info)
        
        info['current_device'] = torch.cuda.current_device()
        info['gpu_memory'] = get_gpu_memory_info()
    
    return info

def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        # Get initial memory
        initial_gpu = get_gpu_memory()
        initial_cpu = get_cpu_memory_info()['used']
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_gpu = get_gpu_memory()
        final_cpu = get_cpu_memory_info()['used']
        
        # Calculate differences
        gpu_diff = final_gpu - initial_gpu
        cpu_diff = final_cpu - initial_cpu
        
        logger.info(f"{func.__name__} memory usage - GPU: {gpu_diff:.2f}MB, CPU: {cpu_diff:.2f}MB")
        
        return result
    return wrapper

def estimate_model_memory(num_parameters: int, dtype: torch.dtype = torch.float32) -> float:
    """Estimate memory required for a model in MB."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)
    
    # Model parameters + gradients + optimizer states (approx 2x for Adam)
    total_bytes = num_parameters * bytes_per_param * 3
    return total_bytes / 1024 / 1024  # Convert to MB

def check_memory_requirements(model_size: int, batch_size: int, sequence_length: int, 
                            dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """Check if current system can handle the memory requirements."""
    
    # Estimate model memory
    model_memory = estimate_model_memory(model_size, dtype)
    
    # Estimate activation memory (rough approximation)
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 4)
    
    # Rough estimate: batch_size * seq_len * hidden_dim * num_layers * overhead
    activation_memory = batch_size * sequence_length * 1024 * 12 * bytes_per_element / 1024 / 1024
    
    total_required = model_memory + activation_memory
    
    current_memory = get_gpu_memory_info() if torch.cuda.is_available() else get_cpu_memory_info()
    available = current_memory.get('free', current_memory.get('available', 0))
    
    return {
        'model_memory_mb': model_memory,
        'activation_memory_mb': activation_memory,
        'total_required_mb': total_required,
        'available_mb': available,
        'sufficient': available > total_required * 1.2,  # 20% safety margin
        'utilization_percent': (total_required / available) * 100 if available > 0 else float('inf')
    }

class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.initial_gpu = 0.0
        self.initial_cpu = 0.0
        
    def __enter__(self):
        self.initial_gpu = get_gpu_memory()
        self.initial_cpu = get_cpu_memory_info()['used']
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        final_gpu = get_gpu_memory()
        final_cpu = get_cpu_memory_info()['used']
        
        gpu_diff = final_gpu - self.initial_gpu
        cpu_diff = final_cpu - self.initial_cpu
        
        logger.info(f"{self.name} - GPU memory delta: {gpu_diff:.2f}MB, CPU memory delta: {cpu_diff:.2f}MB")

def setup_distributed_training() -> Tuple[int, int]:
    """Setup distributed training if available."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        
        logger.info(f"Distributed training setup: rank {rank}/{world_size}, local_rank {local_rank}")
        return rank, world_size
    
    return 0, 1

def print_system_info():
    """Print comprehensive system information."""
    info = get_device_info()
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"CPU: {info['cpu_count']} cores ({info['cpu_count_logical']} logical)")
    if info['cpu_freq']:
        print(f"CPU Frequency: {info['cpu_freq']['current']:.2f} MHz")
    
    cpu_mem = info['cpu_memory']
    print(f"CPU Memory: {cpu_mem['used']:.1f}MB / {cpu_mem['total']:.1f}MB ({cpu_mem['percentage']:.1f}%)")
    
    if info['cuda_available']:
        print(f"\nGPU: {info['cuda_device_count']} device(s) available")
        for device in info['cuda_devices']:
            print(f"  Device {device['device_id']}: {device['name']}")
            print(f"    Memory: {device['total_memory']:.1f}MB")
            print(f"    Compute: {device['major']}.{device['minor']}")
        
        gpu_mem = info['gpu_memory']
        print(f"\nCurrent GPU Memory: {gpu_mem['allocated']:.1f}MB / {gpu_mem['total']:.1f}MB ({gpu_mem['utilization']:.1f}%)")
    else:
        print("\nGPU: Not available")
    
    print("=" * 60)
