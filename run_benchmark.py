#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Benchmark Runner
Main script to run all benchmarks
"""

import logging
from benchmark_core import EnhancedCEPBenchmark
from benchmark_energy import EnergyBenchmark
from benchmark_visualization import BenchmarkVisualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main benchmark function
    """
    print("🎯 Enhanced CEP-EIT-P Comprehensive Benchmark Suite")
    print("=" * 60)
    
    try:
        # Initialize core benchmark
        benchmark = EnhancedCEPBenchmark()
        
        # Run core benchmarks
        benchmark.benchmark_inference_speed()
        benchmark.benchmark_memory_usage()
        benchmark.benchmark_consciousness_accuracy()
        
        # Run energy benchmarks
        energy_benchmark = EnergyBenchmark(benchmark.results)
        energy_benchmark.benchmark_energy_efficiency()
        energy_benchmark.benchmark_scalability()
        
        # Create visualizations and generate report
        visualizer = BenchmarkVisualization(benchmark.results)
        visualizer.create_visualizations()
        visualizer.generate_report()
        
        print("\n🎉 Benchmark suite completed successfully!")
        print("Enhanced CEP-EIT-P performance analysis is complete!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        logger.error(f"Benchmark error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
