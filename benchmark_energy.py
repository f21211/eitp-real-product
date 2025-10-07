#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Energy Benchmark
Energy efficiency and scalability benchmarking
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List
import logging
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyBenchmark:
    """
    Energy efficiency and scalability benchmarking
    """
    
    def __init__(self, results_dict):
        self.results = results_dict
    
    def benchmark_energy_efficiency(self):
        """
        Benchmark energy efficiency across different configurations
        """
        print("\n⚡ Energy Efficiency Benchmark")
        print("=" * 50)
        
        # Test different input complexities
        complexities = [0.1, 0.3, 0.5, 0.7, 0.9]
        batch_sizes = [1, 8, 16, 32, 64]
        
        energy_results = {}
        
        for complexity in complexities:
            print(f"\n📊 Testing with complexity: {complexity}")
            
            cep_params = CEPParameters(
                fractal_dimension=2.7,
                complexity_coefficient=0.8,
                critical_temperature=1.0
            )
            
            model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=cep_params
            )
            
            complexity_results = {}
            
            for batch_size in batch_sizes:
                x = torch.randn(batch_size, 784) * complexity
                output, metrics = model(x)
                
                cep_energies = metrics['cep_energies']
                iem_energy = metrics['iem_energy']
                
                # Calculate energy efficiency
                total_energy = cep_energies['total_energy']
                mass_energy = cep_energies['mass_energy']
                efficiency = total_energy / (mass_energy + 1e-8) if mass_energy != 0 else 0
                
                complexity_results[batch_size] = {
                    'total_energy': total_energy,
                    'mass_energy': mass_energy,
                    'field_energy': cep_energies['field_energy'],
                    'entropy_energy': cep_energies['entropy_energy'],
                    'iem_energy': iem_energy.mean().item() if hasattr(iem_energy, 'mean') else iem_energy,
                    'efficiency': efficiency
                }
            
            energy_results[complexity] = complexity_results
            
            # Print summary for this complexity
            avg_efficiency = np.mean([r['efficiency'] for r in complexity_results.values()])
            print(f"    Average Efficiency: {avg_efficiency:.6f}")
        
        self.results['energy_efficiency'] = energy_results
    
    def benchmark_scalability(self):
        """
        Benchmark scalability with increasing model size
        """
        print("\n📈 Scalability Benchmark")
        print("=" * 50)
        
        # Test different model sizes
        model_sizes = [
            {'input_dim': 256, 'hidden_dims': [128, 64], 'output_dim': 10, 'name': 'Tiny'},
            {'input_dim': 512, 'hidden_dims': [256, 128], 'output_dim': 20, 'name': 'Small'},
            {'input_dim': 1024, 'hidden_dims': [512, 256], 'output_dim': 50, 'name': 'Medium'},
            {'input_dim': 2048, 'hidden_dims': [1024, 512], 'output_dim': 100, 'name': 'Large'}
        ]
        
        scalability_results = {}
        
        for size_config in model_sizes:
            print(f"\n📊 Testing {size_config['name']} Model:")
            
            cep_params = CEPParameters(
                fractal_dimension=2.7,
                complexity_coefficient=0.8,
                critical_temperature=1.0
            )
            
            model = EnhancedCEPEITP(
                input_dim=size_config['input_dim'],
                hidden_dims=size_config['hidden_dims'],
                output_dim=size_config['output_dim'],
                cep_params=cep_params
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Test inference time
            x = torch.randn(32, size_config['input_dim'])
            
            times = []
            for _ in range(5):
                start_time = time.time()
                output, metrics = model(x)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = 32 / avg_time
            
            print(f"    Parameters: {total_params:,}")
            print(f"    Inference Time: {avg_time:.4f}s")
            print(f"    Throughput: {throughput:.0f} samples/s")
            
            scalability_results[size_config['name']] = {
                'parameters': total_params,
                'inference_time': avg_time,
                'throughput': throughput,
                'consciousness_level': metrics['consciousness_metrics'].consciousness_level
            }
        
        self.results['scalability'] = scalability_results
