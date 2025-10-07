#!/usr/bin/env python3
"""
Test script for Enhanced CEP-EIT-P API
"""

import requests
import json
import numpy as np

def test_enhanced_api():
    """Test all enhanced API endpoints"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Enhanced CEP-EIT-P API")
    print("=" * 40)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data['status']}")
            print(f"   📊 Model initialized: {data['model_initialized']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/model_info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model: {data['model_info']['architecture']}")
            print(f"   📊 Parameters: {data['model_info']['total_parameters']:,}")
            print(f"   🧠 CEP params: {data['model_info']['cep_parameters']}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model info error: {e}")
    
    # Test 3: Inference
    print("\n3. Testing inference endpoint...")
    try:
        # Create test input
        test_input = np.random.randn(784).tolist()
        
        response = requests.post(
            f"{base_url}/api/inference",
            json={"input": test_input},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Inference successful")
            print(f"   📊 Output shape: {len(data['output'])} x {len(data['output'][0])}")
            print(f"   🧠 Consciousness level: {data['consciousness_metrics']['level']}/4")
            print(f"   📐 Fractal dimension: {data['consciousness_metrics']['fractal_dimension']:.3f}")
            print(f"   ⚡ Inference time: {data['inference_time']:.4f}s")
        else:
            print(f"   ❌ Inference failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Inference error: {e}")
    
    # Test 4: Consciousness analysis
    print("\n4. Testing consciousness endpoint...")
    try:
        response = requests.get(f"{base_url}/api/consciousness")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Consciousness analysis successful")
            print(f"   📊 Avg level: {data['analysis']['avg_consciousness_level']:.2f}")
            print(f"   📈 Max level: {data['analysis']['max_consciousness_level']}")
        else:
            print(f"   ❌ Consciousness analysis failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Consciousness analysis error: {e}")
    
    # Test 5: Energy analysis
    print("\n5. Testing energy analysis endpoint...")
    try:
        test_input = np.random.randn(784).tolist()
        
        response = requests.post(
            f"{base_url}/api/energy_analysis",
            json={"input": test_input},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Energy analysis successful")
            print(f"   ⚡ Total energy: {data['energy_analysis']['cep_energies']['total_energy']:.6f}")
            print(f"   🔋 Efficiency: {data['energy_analysis']['energy_efficiency']:.6f}")
        else:
            print(f"   ❌ Energy analysis failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Energy analysis error: {e}")
    
    # Test 6: Performance metrics
    print("\n6. Testing performance endpoint...")
    try:
        response = requests.get(f"{base_url}/api/performance")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Performance metrics successful")
            print(f"   📊 Total requests: {data['performance']['total_requests']}")
            print(f"   ⚡ Avg inference time: {data['performance']['avg_inference_time']:.4f}s")
        else:
            print(f"   ❌ Performance metrics failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Performance metrics error: {e}")
    
    print("\n🎉 Enhanced CEP-EIT-P API testing completed!")

if __name__ == "__main__":
    test_enhanced_api()
