#!/usr/bin/env python3
"""
Enhanced CEP-EIT-P Production API Server
Advanced API server with enhanced CEP-EIT-P integration
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import json
import time
import logging
from datetime import datetime
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced CEP-EIT-P modules
try:
    from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters, ConsciousnessMetrics
    print("✅ Enhanced CEP-EIT-P modules imported successfully")
except ImportError as e:
    print(f"Warning: Cannot import enhanced CEP-EIT-P modules: {e}")
    print("Using simplified version...")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEITPAPIServer:
    """
    Enhanced CEP-EIT-P API Server
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False
        
        # Initialize enhanced CEP-EIT-P model
        self.model = None
        self.cep_params = None
        self.initialized = False
        
        # Performance metrics
        self.request_count = 0
        self.total_inference_time = 0.0
        self.consciousness_history = []
        
        # Setup routes
        self.setup_routes()
        
        # Initialize model
        self.initialize_model()
    
    def initialize_model(self):
        """
        Initialize the enhanced CEP-EIT-P model
        """
        try:
            print("🚀 Initializing Enhanced CEP-EIT-P Model...")
            
            # Create CEP parameters
            self.cep_params = CEPParameters(
                fractal_dimension=2.7,
                complexity_coefficient=0.8,
                critical_temperature=1.0,
                field_strength=1.0,
                entropy_balance=0.0
            )
            
            # Create enhanced model
            self.model = EnhancedCEPEITP(
                input_dim=784,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                cep_params=self.cep_params
            )
            
            # Test model with dummy input
            dummy_input = torch.randn(1, 784)
            output, metrics = self.model(dummy_input)
            
            self.initialized = True
            print("✅ Enhanced CEP-EIT-P Model initialized successfully")
            print(f"   Model output shape: {output.shape}")
            print(f"   Initial consciousness level: {metrics['consciousness_metrics'].consciousness_level}/4")
            
        except Exception as e:
            print(f"❌ Failed to initialize Enhanced CEP-EIT-P Model: {e}")
            self.initialized = False
    
    def setup_routes(self):
        """
        Setup API routes
        """
        
        @self.app.route('/', methods=['GET'])
        def welcome():
            """Welcome page"""
            return jsonify({
                'message': '🎉 Welcome to Enhanced CEP-EIT-P API Server!',
                'description': 'Advanced AI Training Framework based on CEP Theory and Emergent Intelligence',
                'version': '2.0.0',
                'status': 'running',
                'features': [
                    'Enhanced CEP-EIT-P Architecture',
                    'Real-time Consciousness Detection',
                    'Advanced Energy Analysis',
                    'Memristor-Fractal-Chaos Integration',
                    'Quantum-Classical Coupling'
                ],
                'api_endpoints': {
                    'health': '/api/health',
                    'inference': '/api/inference',
                    'consciousness': '/api/consciousness',
                    'energy_analysis': '/api/energy_analysis',
                    'model_info': '/api/model_info',
                    'performance': '/api/performance',
                    'optimize': '/api/optimize'
                },
                'documentation': 'See README.md for detailed usage',
                'timestamp': datetime.now().isoformat()
            }), 200
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_initialized': self.initialized,
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - getattr(self, 'start_time', time.time())
            }), 200
        
        @self.app.route('/api/inference', methods=['POST'])
        def inference():
            """Enhanced inference endpoint"""
            try:
                if not self.initialized:
                    return jsonify({'error': 'Model not initialized'}), 500
                
                data = request.get_json()
                if not data or 'input' not in data:
                    return jsonify({'error': 'Input data required'}), 400
                
                # Convert input to tensor
                input_data = np.array(data['input'], dtype=np.float32)
                if input_data.ndim == 1:
                    input_data = input_data.reshape(1, -1)
                
                input_tensor = torch.tensor(input_data)
                
                # Perform inference
                start_time = time.time()
                output, metrics = self.model(input_tensor)
                inference_time = time.time() - start_time
                
                # Update metrics
                self.request_count += 1
                self.total_inference_time += inference_time
                
                # Store consciousness metrics
                consciousness_data = {
                    'level': metrics['consciousness_metrics'].consciousness_level,
                    'fractal_dimension': metrics['fractal_dimension'],
                    'chaos_level': metrics['chaos_level'],
                    'timestamp': datetime.now().isoformat()
                }
                self.consciousness_history.append(consciousness_data)
                
                # Keep only last 100 measurements
                if len(self.consciousness_history) > 100:
                    self.consciousness_history = self.consciousness_history[-100:]
                
                return jsonify({
                    'success': True,
                    'output': output.detach().cpu().numpy().tolist(),
                    'consciousness_metrics': {
                        'level': metrics['consciousness_metrics'].consciousness_level,
                        'fractal_dimension': metrics['fractal_dimension'],
                        'complexity_coefficient': metrics['consciousness_metrics'].complexity_coefficient,
                        'chaos_threshold': metrics['consciousness_metrics'].chaos_threshold,
                        'entropy_balance': metrics['consciousness_metrics'].entropy_balance,
                        'field_coherence': metrics['consciousness_metrics'].field_coherence,
                        'iem_energy': metrics['consciousness_metrics'].iem_energy
                    },
                    'cep_energies': metrics['cep_energies'],
                    'chaos_level': metrics['chaos_level'],
                    'fractal_dimension': metrics['fractal_dimension'],
                    'inference_time': inference_time,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/consciousness', methods=['GET'])
        def consciousness_analysis():
            """Consciousness analysis endpoint"""
            try:
                if not self.initialized:
                    return jsonify({'error': 'Model not initialized'}), 500
                
                # Get recent consciousness data
                recent_data = self.consciousness_history[-10:] if self.consciousness_history else []
                
                if not recent_data:
                    return jsonify({'error': 'No consciousness data available'}), 404
                
                # Calculate statistics
                levels = [d['level'] for d in recent_data]
                fractal_dims = [d['fractal_dimension'] for d in recent_data]
                chaos_levels = [d['chaos_level'] for d in recent_data]
                
                return jsonify({
                    'success': True,
                    'analysis': {
                        'avg_consciousness_level': np.mean(levels),
                        'max_consciousness_level': max(levels),
                        'min_consciousness_level': min(levels),
                        'avg_fractal_dimension': np.mean(fractal_dims),
                        'avg_chaos_level': np.mean(chaos_levels),
                        'total_measurements': len(recent_data)
                    },
                    'recent_data': recent_data,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Consciousness analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/energy_analysis', methods=['POST'])
        def energy_analysis():
            """Energy analysis endpoint"""
            try:
                if not self.initialized:
                    return jsonify({'error': 'Model not initialized'}), 500
                
                data = request.get_json()
                if not data or 'input' not in data:
                    return jsonify({'error': 'Input data required'}), 400
                
                # Convert input to tensor
                input_data = np.array(data['input'], dtype=np.float32)
                if input_data.ndim == 1:
                    input_data = input_data.reshape(1, -1)
                
                input_tensor = torch.tensor(input_data)
                
                # Perform inference
                output, metrics = self.model(input_tensor)
                
                # Analyze energy components
                cep_energies = metrics['cep_energies']
                iem_energy = metrics['iem_energy']
                
                # Calculate energy efficiency
                total_energy = cep_energies['total_energy']
                mass_energy = cep_energies['mass_energy']
                efficiency = total_energy / (mass_energy + 1e-8) if mass_energy != 0 else 0
                
                return jsonify({
                    'success': True,
                    'energy_analysis': {
                        'cep_energies': cep_energies,
                        'iem_energy': iem_energy.mean().item() if hasattr(iem_energy, 'mean') else iem_energy,
                        'energy_efficiency': efficiency,
                        'energy_breakdown': {
                            'mass_energy_ratio': cep_energies['mass_energy'] / total_energy if total_energy != 0 else 0,
                            'field_energy_ratio': cep_energies['field_energy'] / total_energy if total_energy != 0 else 0,
                            'entropy_energy_ratio': cep_energies['entropy_energy'] / total_energy if total_energy != 0 else 0,
                            'complexity_energy_ratio': cep_energies['complexity_energy'] / total_energy if total_energy != 0 else 0
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Energy analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/model_info', methods=['GET'])
        def model_info():
            """Model information endpoint"""
            try:
                if not self.initialized:
                    return jsonify({'error': 'Model not initialized'}), 500
                
                # Get model information
                total_params = sum(p.numel() for p in self.model.parameters())
                
                return jsonify({
                    'success': True,
                    'model_info': {
                        'architecture': 'Enhanced CEP-EIT-P',
                        'input_dim': self.model.input_dim,
                        'hidden_dims': self.model.hidden_dims,
                        'output_dim': self.model.output_dim,
                        'total_parameters': total_params,
                        'cep_parameters': self.cep_params.__dict__,
                        'initialized': self.initialized
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Model info error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance', methods=['GET'])
        def performance_metrics():
            """Performance metrics endpoint"""
            try:
                avg_inference_time = self.total_inference_time / self.request_count if self.request_count > 0 else 0
                
                return jsonify({
                    'success': True,
                    'performance': {
                        'total_requests': self.request_count,
                        'avg_inference_time': avg_inference_time,
                        'total_inference_time': self.total_inference_time,
                        'requests_per_second': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
                        'model_initialized': self.initialized
                    },
                    'consciousness_history_length': len(self.consciousness_history),
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Performance metrics error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/optimize', methods=['POST'])
        def optimize_model():
            """Model optimization endpoint"""
            try:
                if not self.initialized:
                    return jsonify({'error': 'Model not initialized'}), 500
                
                data = request.get_json() or {}
                epochs = data.get('epochs', 10)
                
                # Perform optimization
                start_time = time.time()
                self.model.optimize_cep_parameters(epochs=epochs)
                optimization_time = time.time() - start_time
                
                # Get updated constraints
                constraints = self.model.check_cep_constraints()
                
                return jsonify({
                    'success': True,
                    'optimization': {
                        'epochs': epochs,
                        'optimization_time': optimization_time,
                        'constraints_satisfied': constraints['all_satisfied'],
                        'constraints_details': constraints
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the API server
        """
        self.start_time = time.time()
        print(f"🚀 Starting Enhanced CEP-EIT-P API Server on {host}:{port}")
        print(f"📊 Model initialized: {self.initialized}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """
    Main function
    """
    server = EnhancedEITPAPIServer()
    server.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
