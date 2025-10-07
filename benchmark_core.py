#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Benchmark Core
Core benchmarking functionality
"""

import torch
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCEPBenchmark:
    """
    Enhanced CEP-EIT-P Benchmark Suite
    """
    
    def __init__(self):
        self.results = {}
        self.memory_usage = []
        self.cpu_usage = []
        
    def benchmark_inference_speed(self):
        """
        Benchmark inference speed across different configurations
        """
        print("🚀 Inference Speed Benchmark")
        print("=" * 50)
        
        # Test configurations
        configs = [
            {'input_dim': 784, 'hidden_dims': [512, 256, 128], 'output_dim': 10, 'name': 'Small'},
            {'input_dim': 1024, 'hidden_dims': [768, 512, 256], 'output_dim': 20, 'name': 'Medium'},
            {'input_dim': 2048, 'hidden_dims': [1536, 1024, 512], 'output_dim': 50, 'name': 'Large'}
        ]
        
        batch_sizes = [1, 8, 16, 32, 64]
        
        speed_results = {}
        
        for config in configs:
            print(f"\n📊 Testing {config['name']} Configuration:")
            print(f"  Input: {config['input_dim']}, Hidden: {config['hidden_dims']}, Output: {config['output_dim']}")
            
            # Create model
            cep_params = CEPParameters(
                fractal_dimension=2.7,
                complexity_coefficient=0.8,
                critical_temperature=1.0
            )
            
            model = EnhancedCEPEITP(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims'],
                output_dim=config['output_dim'],
                cep_params=cep_params
            )
            
            config_results = {}
            
            for batch_size in batch_sizes:
                # Warm up
                x_warmup = torch.randn(1, config['input_dim'])
                _ = model(x_warmup)
                
                # Benchmark
                x = torch.randn(batch_size, config['input_dim'])
                
                # Measure time
                times = []
                for _ in range(5):  # 5 iterations for average
                    start_time = time.time()
                    output, metrics = model(x)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                throughput = batch_size / avg_time
                
                print(f"    Batch {batch_size:3d}: {avg_time:.4f}s ± {std_time:.4f}s ({throughput:.0f} samples/s)")
                
                config_results[batch_size] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'throughput': throughput,
                    'consciousness_level': metrics['consciousness_metrics'].consciousness_level
                }
            
            speed_results[config['name']] = config_results
        
        self.results['inference_speed'] = speed_results
    
    def benchmark_memory_usage(self):
        """
        Benchmark memory usage across different model sizes
        """
        print("\n💾 Memory Usage Benchmark")
        print("=" * 50)
        
        configs = [
            {'input_dim': 784, 'hidden_dims': [512, 256, 128], 'output_dim': 10, 'name': 'Small'},
            {'input_dim': 1024, 'hidden_dims': [768, 512, 256], 'output_dim': 20, 'name': 'Medium'},
            {'input_dim': 2048, 'hidden_dims': [1536, 1024, 512], 'output_dim': 50, 'name': 'Large'}
        ]
        
        memory_results = {}
        
        for config in configs:
            print(f"\n📊 Testing {config['name']} Configuration:")
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Measure baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create model
            cep_params = CEPParameters(
                fractal_dimension=2.7,
                complexity_coefficient=0.8,
                critical_temperature=1.0
            )
            
            model = EnhancedCEPEITP(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims'],
                output_dim=config['output_dim'],
                cep_params=cep_params
            )
            
            # Measure model memory
            model_memory = process.memory_info().rss / 1024 / 1024  # MB
            model_size = model_memory - baseline_memory
            
            # Test with different batch sizes
            batch_memory = {}
            for batch_size in [1, 8, 16, 32, 64]:
                x = torch.randn(batch_size, config['input_dim'])
                output, metrics = model(x)
                
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                batch_memory[batch_size] = current_memory - model_memory
                
                print(f"    Batch {batch_size:3d}: {batch_memory[batch_size]:.2f} MB")
            
            memory_results[config['name']] = {
                'model_size': model_size,
                'batch_memory': batch_memory,
                'total_params': sum(p.numel() for p in model.parameters())
            }
            
            print(f"    Model Size: {model_size:.2f} MB")
            print(f"    Total Parameters: {memory_results[config['name']]['total_params']:,}")
        
        self.results['memory_usage'] = memory_results
    
    def benchmark_consciousness_accuracy(self):
        """
        Benchmark consciousness detection accuracy
        """
        print("\n🧠 Consciousness Accuracy Benchmark")
        print("=" * 50)
        
        # Test different CEP parameter configurations
        cep_configs = [
            {'fractal_dimension': 2.5, 'complexity_coefficient': 0.7, 'expected_level': 1, 'name': 'Low'},
            {'fractal_dimension': 2.7, 'complexity_coefficient': 0.8, 'expected_level': 2, 'name': 'Medium'},
            {'fractal_dimension': 2.9, 'complexity_coefficient': 0.9, 'expected_level': 3, 'name': 'High'},
            {'fractal_dimension': 3.0, 'complexity_coefficient': 1.0, 'expected_level': 4, 'name': 'Maximum'}
        ]
        
        accuracy_results = {}
        
        for config in cep_configs:
            print(f"\n📊 Testing {config['name']} Consciousness Configuration:")
            
            cep_params = CEPParameters(
                fractal_dimension=config['fractal_dimension'],
                complexity_coefficient=config['complexity_coefficient'],
                critical_temperature=1.0
            )
            
            model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=cep_params
            )
            
            # Test with multiple samples
            consciousness_levels = []
            constraint_satisfactions = []
            
            for _ in range(50):  # 50 test samples
                x = torch.randn(32, 784)
                output, metrics = model(x)
                
                consciousness_level = metrics['consciousness_metrics'].consciousness_level
                consciousness_levels.append(consciousness_level)
                
                constraints = model.check_cep_constraints()
                constraint_satisfactions.append(constraints['all_satisfied'])
            
            avg_consciousness = np.mean(consciousness_levels)
            std_consciousness = np.std(consciousness_levels)
            constraint_rate = np.mean(constraint_satisfactions)
            
            print(f"    Expected Level: {config['expected_level']}")
            print(f"    Actual Level: {avg_consciousness:.2f} ± {std_consciousness:.2f}")
            print(f"    Constraint Satisfaction: {constraint_rate:.2%}")
            
            accuracy_results[config['name']] = {
                'expected_level': config['expected_level'],
                'actual_level': avg_consciousness,
                'std_level': std_consciousness,
                'constraint_rate': constraint_rate,
                'fractal_dimension': config['fractal_dimension'],
                'complexity_coefficient': config['complexity_coefficient']
            }
        
        self.results['consciousness_accuracy'] = accuracy_results
