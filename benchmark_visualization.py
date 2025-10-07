#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Benchmark Visualization
Visualization and reporting functionality
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import time
from typing import Dict

class BenchmarkVisualization:
    """
    Benchmark visualization and reporting
    """
    
    def __init__(self, results_dict):
        self.results = results_dict
    
    def create_visualizations(self):
        """
        Create comprehensive benchmark visualizations
        """
        print("\n📊 Creating Benchmark Visualizations...")
        
        # 1. Inference Speed Comparison
        if 'inference_speed' in self.results:
            self._create_inference_plots()
        
        # 2. Memory Usage Analysis
        if 'memory_usage' in self.results:
            self._create_memory_plots()
        
        # 3. Consciousness Accuracy Analysis
        if 'consciousness_accuracy' in self.results:
            self._create_consciousness_plots()
        
        # 4. Energy Efficiency Analysis
        if 'energy_efficiency' in self.results:
            self._create_energy_plots()
        
        # 5. Scalability Analysis
        if 'scalability' in self.results:
            self._create_scalability_plots()
    
    def _create_inference_plots(self):
        """Create inference speed plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput vs Batch Size
        for config_name, config_data in self.results['inference_speed'].items():
            batch_sizes = list(config_data.keys())
            throughputs = [config_data[bs]['throughput'] for bs in batch_sizes]
            ax1.plot(batch_sizes, throughputs, 'o-', label=config_name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (samples/s)')
        ax1.set_title('Inference Throughput vs Batch Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Consciousness Level vs Configuration
        config_names = list(self.results['inference_speed'].keys())
        consciousness_levels = [self.results['inference_speed'][name][32]['consciousness_level'] for name in config_names]
        
        ax2.bar(config_names, consciousness_levels, color=['blue', 'green', 'orange'])
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('Consciousness Level')
        ax2.set_title('Consciousness Level by Model Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_cep_inference_benchmark.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved: enhanced_cep_inference_benchmark.png")
    
    def _create_memory_plots(self):
        """Create memory usage plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        config_names = list(self.results['memory_usage'].keys())
        model_sizes = [self.results['memory_usage'][name]['model_size'] for name in config_names]
        total_params = [self.results['memory_usage'][name]['total_params'] for name in config_names]
        
        # Model Size vs Configuration
        ax1.bar(config_names, model_sizes, color=['blue', 'green', 'orange'])
        ax1.set_xlabel('Model Configuration')
        ax1.set_ylabel('Model Size (MB)')
        ax1.set_title('Model Memory Usage')
        ax1.grid(True, alpha=0.3)
        
        # Parameters vs Configuration
        ax2.bar(config_names, total_params, color=['blue', 'green', 'orange'])
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('Total Parameters')
        ax2.set_title('Model Parameter Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_cep_memory_benchmark.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved: enhanced_cep_memory_benchmark.png")
    
    def _create_consciousness_plots(self):
        """Create consciousness accuracy plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        config_names = list(self.results['consciousness_accuracy'].keys())
        expected_levels = [self.results['consciousness_accuracy'][name]['expected_level'] for name in config_names]
        actual_levels = [self.results['consciousness_accuracy'][name]['actual_level'] for name in config_names]
        std_levels = [self.results['consciousness_accuracy'][name]['std_level'] for name in config_names]
        constraint_rates = [self.results['consciousness_accuracy'][name]['constraint_rate'] for name in config_names]
        
        # Expected vs Actual Consciousness Levels
        x = np.arange(len(config_names))
        width = 0.35
        
        ax1.bar(x - width/2, expected_levels, width, label='Expected', alpha=0.8)
        ax1.bar(x + width/2, actual_levels, width, label='Actual', alpha=0.8, yerr=std_levels, capsize=5)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Consciousness Level')
        ax1.set_title('Expected vs Actual Consciousness Levels')
        ax1.set_xticks(x)
        ax1.set_xticklabels(config_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Constraint Satisfaction Rate
        ax2.bar(config_names, constraint_rates, color=['red', 'orange', 'green', 'blue'])
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Constraint Satisfaction Rate')
        ax2.set_title('CEP Constraint Satisfaction')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('enhanced_cep_consciousness_benchmark.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved: enhanced_cep_consciousness_benchmark.png")
    
    def _create_energy_plots(self):
        """Create energy efficiency plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        complexities = list(self.results['energy_efficiency'].keys())
        batch_sizes = [1, 8, 16, 32, 64]
        
        # Energy vs Complexity
        for batch_size in batch_sizes:
            if batch_size in self.results['energy_efficiency'][complexities[0]]:
                energies = [self.results['energy_efficiency'][c][batch_size]['total_energy'] for c in complexities]
                ax1.plot(complexities, energies, 'o-', label=f'Batch {batch_size}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Input Complexity')
        ax1.set_ylabel('Total Energy (J)')
        ax1.set_title('Energy Consumption vs Input Complexity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency vs Batch Size
        avg_efficiencies = []
        for batch_size in batch_sizes:
            if batch_size in self.results['energy_efficiency'][complexities[0]]:
                effs = [self.results['energy_efficiency'][c][batch_size]['efficiency'] for c in complexities]
                avg_efficiencies.append(np.mean(effs))
            else:
                avg_efficiencies.append(0)
        
        ax2.plot(batch_sizes, avg_efficiencies, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Average Energy Efficiency')
        ax2.set_title('Energy Efficiency vs Batch Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('enhanced_cep_energy_benchmark.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved: enhanced_cep_energy_benchmark.png")
    
    def _create_scalability_plots(self):
        """Create scalability plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        config_names = list(self.results['scalability'].keys())
        parameters = [self.results['scalability'][name]['parameters'] for name in config_names]
        throughputs = [self.results['scalability'][name]['throughput'] for name in config_names]
        consciousness_levels = [self.results['scalability'][name]['consciousness_level'] for name in config_names]
        
        # Parameters vs Throughput
        ax1.scatter(parameters, throughputs, s=100, c=consciousness_levels, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('Throughput (samples/s)')
        ax1.set_title('Scalability: Parameters vs Throughput')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Add colorbar for consciousness levels
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Consciousness Level')
        
        # Consciousness Level vs Model Size
        ax2.bar(config_names, consciousness_levels, color=['blue', 'green', 'orange', 'red'])
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('Consciousness Level')
        ax2.set_title('Consciousness Level vs Model Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 4)
        
        plt.tight_layout()
        plt.savefig('enhanced_cep_scalability_benchmark.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved: enhanced_cep_scalability_benchmark.png")
    
    def generate_report(self):
        """
        Generate comprehensive benchmark report
        """
        print("\n📋 Generating Benchmark Report...")
        
        # Calculate summary statistics
        summary = {
            'total_benchmarks': len(self.results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'framework': 'Enhanced CEP-EIT-P',
            'version': '2.0.0'
        }
        
        # Add performance summaries
        if 'inference_speed' in self.results:
            max_throughput = 0
            best_config = None
            for config_name, config_data in self.results['inference_speed'].items():
                for batch_size, data in config_data.items():
                    if data['throughput'] > max_throughput:
                        max_throughput = data['throughput']
                        best_config = f"{config_name} (batch {batch_size})"
            
            summary['inference_performance'] = {
                'max_throughput': max_throughput,
                'best_configuration': best_config
            }
        
        if 'consciousness_accuracy' in self.results:
            avg_accuracy = np.mean([data['constraint_rate'] for data in self.results['consciousness_accuracy'].values()])
            summary['consciousness_performance'] = {
                'average_constraint_satisfaction': avg_accuracy
            }
        
        if 'scalability' in self.results:
            max_consciousness = max([data['consciousness_level'] for data in self.results['scalability'].values()])
            summary['scalability_performance'] = {
                'max_consciousness_level': max_consciousness
            }
        
        # Create comprehensive report
        report = {
            'summary': summary,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open('enhanced_cep_benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("  ✅ Saved: enhanced_cep_benchmark_report.json")
        
        # Print summary
        print("\n📊 Benchmark Summary:")
        print(f"  🧪 Total Benchmarks: {summary['total_benchmarks']}")
        if 'inference_performance' in summary:
            print(f"  🚀 Max Throughput: {summary['inference_performance']['max_throughput']:.0f} samples/s")
            print(f"  🏆 Best Configuration: {summary['inference_performance']['best_configuration']}")
        if 'consciousness_performance' in summary:
            print(f"  🧠 Avg Constraint Satisfaction: {summary['consciousness_performance']['average_constraint_satisfaction']:.2%}")
        if 'scalability_performance' in summary:
            print(f"  📈 Max Consciousness Level: {summary['scalability_performance']['max_consciousness_level']}")
        
        print(f"  📁 Generated Files:")
        print(f"    - enhanced_cep_inference_benchmark.png")
        print(f"    - enhanced_cep_memory_benchmark.png")
        print(f"    - enhanced_cep_consciousness_benchmark.png")
        print(f"    - enhanced_cep_energy_benchmark.png")
        print(f"    - enhanced_cep_scalability_benchmark.png")
        print(f"    - enhanced_cep_benchmark_report.json")
    
    def _generate_recommendations(self):
        """Generate performance recommendations"""
        recommendations = []
        
        if 'inference_speed' in self.results:
            for config_name, config_data in self.results['inference_speed'].items():
                best_batch = max(config_data.keys(), key=lambda x: config_data[x]['throughput'])
                best_throughput = config_data[best_batch]['throughput']
                recommendations.append({
                    'type': 'inference_optimization',
                    'configuration': config_name,
                    'recommendation': f"Use batch size {best_batch} for optimal throughput ({best_throughput:.0f} samples/s)"
                })
        
        if 'consciousness_accuracy' in self.results:
            best_config = max(self.results['consciousness_accuracy'].items(), 
                            key=lambda x: x[1]['constraint_rate'])
            recommendations.append({
                'type': 'consciousness_optimization',
                'recommendation': f"Use {best_config[0]} configuration for best consciousness detection"
            })
        
        return recommendations
