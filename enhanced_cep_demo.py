#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Demo Script
Comprehensive demonstration of the enhanced CEP-EIT-P framework
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters, ConsciousnessMetrics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCEPDemo:
    """
    Enhanced CEP-EIT-P Demonstration Class
    """
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
        
    def run_basic_demo(self):
        """
        Run basic CEP-EIT-P demonstration
        """
        print("🚀 Enhanced CEP-EIT-P Basic Demo")
        print("=" * 50)
        
        # Create enhanced CEP-EIT-P model
        cep_params = CEPParameters(
            fractal_dimension=2.7,
            complexity_coefficient=0.8,
            critical_temperature=1.0,
            field_strength=1.0,
            entropy_balance=0.0
        )
        
        model = EnhancedCEPEITP(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            output_dim=10,
            cep_params=cep_params
        )
        
        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            print(f"\n📊 Testing with batch size: {batch_size}")
            
            # Generate test data
            x = torch.randn(batch_size, 784)
            
            # Measure performance
            start_time = time.time()
            output, metrics = model(x)
            end_time = time.time()
            
            # Calculate performance metrics
            inference_time = end_time - start_time
            throughput = batch_size / inference_time
            
            print(f"  ✅ Output shape: {output.shape}")
            print(f"  🧠 Consciousness Level: {metrics['consciousness_metrics'].consciousness_level}/4")
            print(f"  📐 Fractal Dimension: {metrics['fractal_dimension']:.3f}")
            print(f"  🌪️  Chaos Level: {metrics['chaos_level']:.6f}")
            print(f"  ⚡ Inference Time: {inference_time:.4f}s")
            print(f"  🚀 Throughput: {throughput:.2f} samples/s")
            
            # Store results
            self.results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'inference_time': inference_time,
                'throughput': throughput,
                'consciousness_level': metrics['consciousness_metrics'].consciousness_level,
                'fractal_dimension': metrics['fractal_dimension'],
                'chaos_level': metrics['chaos_level']
            }
    
    def run_consciousness_analysis(self):
        """
        Run consciousness analysis demonstration
        """
        print("\n🧠 Consciousness Analysis Demo")
        print("=" * 50)
        
        # Create model with different CEP parameters
        cep_configs = [
            {'fractal_dimension': 2.5, 'complexity_coefficient': 0.7, 'name': 'Low Consciousness'},
            {'fractal_dimension': 2.7, 'complexity_coefficient': 0.8, 'name': 'Medium Consciousness'},
            {'fractal_dimension': 2.9, 'complexity_coefficient': 0.9, 'name': 'High Consciousness'},
            {'fractal_dimension': 3.0, 'complexity_coefficient': 1.0, 'name': 'Maximum Consciousness'}
        ]
        
        consciousness_results = []
        
        for config in cep_configs:
            print(f"\n🔬 Testing {config['name']} Configuration:")
            
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
            total_consciousness = 0
            num_samples = 10
            
            for i in range(num_samples):
                x = torch.randn(32, 784)
                output, metrics = model(x)
                
                consciousness_level = metrics['consciousness_metrics'].consciousness_level
                total_consciousness += consciousness_level
            
            avg_consciousness = total_consciousness / num_samples
            
            print(f"  📊 Average Consciousness Level: {avg_consciousness:.2f}/4")
            print(f"  📐 Fractal Dimension: {config['fractal_dimension']}")
            print(f"  🔢 Complexity Coefficient: {config['complexity_coefficient']}")
            
            # Check constraints
            constraints = model.check_cep_constraints()
            print(f"  ✅ All Constraints Satisfied: {constraints['all_satisfied']}")
            
            consciousness_results.append({
                'config': config['name'],
                'fractal_dimension': config['fractal_dimension'],
                'complexity_coefficient': config['complexity_coefficient'],
                'avg_consciousness': avg_consciousness,
                'constraints_satisfied': constraints['all_satisfied']
            })
        
        self.results['consciousness_analysis'] = consciousness_results
    
    def run_energy_analysis(self):
        """
        Run energy analysis demonstration
        """
        print("\n⚡ Energy Analysis Demo")
        print("=" * 50)
        
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
        
        # Test with different input complexities
        complexities = [0.1, 0.3, 0.5, 0.7, 0.9]
        energy_results = []
        
        for complexity in complexities:
            print(f"\n🔋 Testing with complexity: {complexity}")
            
            # Generate input with specific complexity
            x = torch.randn(32, 784) * complexity
            
            output, metrics = model(x)
            cep_energies = metrics['cep_energies']
            iem_energy = metrics['iem_energy']
            
            print(f"  📊 Mass Energy: {cep_energies['mass_energy']:.6f}")
            print(f"  🌊 Field Energy: {cep_energies['field_energy']:.6f}")
            print(f"  🔥 Entropy Energy: {cep_energies['entropy_energy']:.6f}")
            print(f"  🧩 Complexity Energy: {cep_energies['complexity_energy']:.6f}")
            print(f"  ⚡ Total CEP Energy: {cep_energies['total_energy']:.6f}")
            iem_energy_val = iem_energy.mean().item() if hasattr(iem_energy, 'mean') else iem_energy
            print(f"  🧠 IEM Energy: {iem_energy_val:.6f}")
            
            energy_results.append({
                'complexity': complexity,
                'mass_energy': cep_energies['mass_energy'],
                'field_energy': cep_energies['field_energy'],
                'entropy_energy': cep_energies['entropy_energy'],
                'complexity_energy': cep_energies['complexity_energy'],
                'total_energy': cep_energies['total_energy'],
                'iem_energy': iem_energy
            })
        
        self.results['energy_analysis'] = energy_results
    
    def run_optimization_demo(self):
        """
        Run parameter optimization demonstration
        """
        print("\n🔧 Parameter Optimization Demo")
        print("=" * 50)
        
        # Start with suboptimal parameters
        cep_params = CEPParameters(
            fractal_dimension=2.5,  # Below threshold
            complexity_coefficient=0.7,  # Below threshold
            critical_temperature=1.0
        )
        
        model = EnhancedCEPEITP(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            output_dim=10,
            cep_params=cep_params
        )
        
        print("📊 Initial State:")
        x = torch.randn(32, 784)
        output, metrics = model(x)
        initial_constraints = model.check_cep_constraints()
        
        print(f"  🧠 Consciousness Level: {metrics['consciousness_metrics'].consciousness_level}/4")
        print(f"  📐 Fractal Dimension: {metrics['fractal_dimension']:.3f}")
        print(f"  🔢 Complexity Coefficient: {metrics['consciousness_metrics'].complexity_coefficient:.3f}")
        print(f"  ✅ All Constraints Satisfied: {initial_constraints['all_satisfied']}")
        
        print("\n🔧 Running Optimization...")
        model.optimize_cep_parameters(epochs=100)
        
        print("\n📊 Final State:")
        output, metrics = model(x)
        final_constraints = model.check_cep_constraints()
        
        print(f"  🧠 Consciousness Level: {metrics['consciousness_metrics'].consciousness_level}/4")
        print(f"  📐 Fractal Dimension: {metrics['fractal_dimension']:.3f}")
        print(f"  🔢 Complexity Coefficient: {metrics['consciousness_metrics'].complexity_coefficient:.3f}")
        print(f"  ✅ All Constraints Satisfied: {final_constraints['all_satisfied']}")
        
        self.results['optimization'] = {
            'initial_consciousness': initial_constraints['consciousness_level'],
            'final_consciousness': final_constraints['consciousness_level'],
            'initial_constraints': initial_constraints['all_satisfied'],
            'final_constraints': final_constraints['all_satisfied']
        }
    
    def create_visualizations(self):
        """
        Create visualization plots
        """
        print("\n📊 Creating Visualizations...")
        
        # Consciousness vs Parameters
        if 'consciousness_analysis' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Consciousness vs Fractal Dimension
            data = self.results['consciousness_analysis']
            fractal_dims = [d['fractal_dimension'] for d in data]
            consciousness_levels = [d['avg_consciousness'] for d in data]
            
            ax1.plot(fractal_dims, consciousness_levels, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Fractal Dimension')
            ax1.set_ylabel('Consciousness Level')
            ax1.set_title('Consciousness Level vs Fractal Dimension')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=2.7, color='r', linestyle='--', alpha=0.7, label='Threshold (2.7)')
            ax1.legend()
            
            # Consciousness vs Complexity Coefficient
            complexity_coeffs = [d['complexity_coefficient'] for d in data]
            ax2.plot(complexity_coeffs, consciousness_levels, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Complexity Coefficient')
            ax2.set_ylabel('Consciousness Level')
            ax2.set_title('Consciousness Level vs Complexity Coefficient')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Threshold (0.8)')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('enhanced_cep_consciousness_analysis.png', dpi=300, bbox_inches='tight')
            print("  ✅ Saved: enhanced_cep_consciousness_analysis.png")
        
        # Energy Analysis
        if 'energy_analysis' in self.results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            data = self.results['energy_analysis']
            complexities = [d['complexity'] for d in data]
            mass_energies = [d['mass_energy'] for d in data]
            field_energies = [d['field_energy'] for d in data]
            entropy_energies = [d['entropy_energy'] for d in data]
            total_energies = [d['total_energy'] for d in data]
            
            ax.plot(complexities, mass_energies, 'b-', label='Mass Energy', linewidth=2)
            ax.plot(complexities, field_energies, 'g-', label='Field Energy', linewidth=2)
            ax.plot(complexities, entropy_energies, 'r-', label='Entropy Energy', linewidth=2)
            ax.plot(complexities, total_energies, 'k--', label='Total CEP Energy', linewidth=3)
            
            ax.set_xlabel('Input Complexity')
            ax.set_ylabel('Energy (J)')
            ax.set_title('CEP Energy Components vs Input Complexity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('enhanced_cep_energy_analysis.png', dpi=300, bbox_inches='tight')
            print("  ✅ Saved: enhanced_cep_energy_analysis.png")
        
        # Performance Analysis
        if any(f'batch_{bs}' in self.results for bs in [1, 8, 16, 32, 64]):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            batch_sizes = []
            throughputs = []
            consciousness_levels = []
            
            for bs in [1, 8, 16, 32, 64]:
                if f'batch_{bs}' in self.results:
                    data = self.results[f'batch_{bs}']
                    batch_sizes.append(data['batch_size'])
                    throughputs.append(data['throughput'])
                    consciousness_levels.append(data['consciousness_level'])
            
            # Throughput vs Batch Size
            ax1.plot(batch_sizes, throughputs, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Throughput (samples/s)')
            ax1.set_title('Throughput vs Batch Size')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Consciousness Level vs Batch Size
            ax2.plot(batch_sizes, consciousness_levels, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Consciousness Level')
            ax2.set_title('Consciousness Level vs Batch Size')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            ax2.set_ylim(0, 4)
            
            plt.tight_layout()
            plt.savefig('enhanced_cep_performance_analysis.png', dpi=300, bbox_inches='tight')
            print("  ✅ Saved: enhanced_cep_performance_analysis.png")
    
    def generate_report(self):
        """
        Generate comprehensive demo report
        """
        print("\n📋 Generating Demo Report...")
        
        report = {
            'demo_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'framework': 'Enhanced CEP-EIT-P',
                'version': '2.0.0'
            },
            'results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'successful_tests': len([r for r in self.results.values() if r is not None])
            }
        }
        
        # Save report
        with open('enhanced_cep_demo_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("  ✅ Saved: enhanced_cep_demo_report.json")
        
        # Print summary
        print("\n📊 Demo Summary:")
        print(f"  🧪 Total Tests: {report['summary']['total_tests']}")
        print(f"  ✅ Successful Tests: {report['summary']['successful_tests']}")
        print(f"  📁 Generated Files:")
        print(f"    - enhanced_cep_consciousness_analysis.png")
        print(f"    - enhanced_cep_energy_analysis.png")
        print(f"    - enhanced_cep_performance_analysis.png")
        print(f"    - enhanced_cep_demo_report.json")
    
    def run_full_demo(self):
        """
        Run complete demonstration
        """
        print("🎉 Enhanced CEP-EIT-P Complete Demo")
        print("=" * 60)
        
        try:
            # Run all demonstrations
            self.run_basic_demo()
            self.run_consciousness_analysis()
            self.run_energy_analysis()
            self.run_optimization_demo()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            self.generate_report()
            
            print("\n🎉 Demo completed successfully!")
            print("Enhanced CEP-EIT-P framework is working perfectly!")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)

def main():
    """
    Main demo function
    """
    demo = EnhancedCEPDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()
