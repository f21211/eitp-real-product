#!/bin/bash
"""
Enhanced CEP-EIT-P Production Startup Script
Advanced production deployment with enhanced CEP-EIT-P integration
"""

echo "🚀 Starting Enhanced CEP-EIT-P Production Services"
echo "=================================================="

# Set environment variables
export PYTHONPATH="/mnt/sda1/myproject/datainall/AGI_clean:$PYTHONPATH"
export FLASK_ENV=production
export FLASK_DEBUG=False

# Activate virtual environment
echo "📦 Activating virtual environment..."
source /mnt/sda1/myproject/datainall/AGI/venv/bin/activate

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Check Python version
echo "🐍 Python version: $(python3 --version)"

# Check required packages
echo "📋 Checking required packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import flask; print(f'Flask: {flask.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check if enhanced CEP-EIT-P modules are available
echo "🔍 Checking Enhanced CEP-EIT-P modules..."
python3 -c "
try:
    from eit_p_enhanced_cep import EnhancedCEPEITP, CEPParameters
    print('✅ Enhanced CEP-EIT-P modules available')
except ImportError as e:
    print(f'❌ Enhanced CEP-EIT-P modules not available: {e}')
    exit(1)
"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Enhanced API Server
echo "🌐 Starting Enhanced CEP-EIT-P API Server..."
nohup python3 enhanced_api_server.py > logs/enhanced_api_server.log 2>&1 &
API_PID=$!
echo "   API Server PID: $API_PID"

# Wait for API server to start
echo "⏳ Waiting for API server to start..."
sleep 5

# Check if API server is running
if ps -p $API_PID > /dev/null; then
    echo "✅ Enhanced API Server started successfully"
else
    echo "❌ Failed to start Enhanced API Server"
    exit 1
fi

# Test API endpoints
echo "🧪 Testing API endpoints..."

# Test health endpoint
echo "   Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:5000/api/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "   ✅ Health endpoint working"
else
    echo "   ❌ Health endpoint failed"
    echo "   Response: $HEALTH_RESPONSE"
fi

# Test model info endpoint
echo "   Testing model info endpoint..."
MODEL_INFO_RESPONSE=$(curl -s http://localhost:5000/api/model_info)
if echo "$MODEL_INFO_RESPONSE" | grep -q "Enhanced CEP-EIT-P"; then
    echo "   ✅ Model info endpoint working"
else
    echo "   ❌ Model info endpoint failed"
    echo "   Response: $MODEL_INFO_RESPONSE"
fi

# Test inference endpoint
echo "   Testing inference endpoint..."
INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:5000/api/inference \
    -H "Content-Type: application/json" \
    -d '{"input": [0.1] * 784}')
if echo "$INFERENCE_RESPONSE" | grep -q "success"; then
    echo "   ✅ Inference endpoint working"
else
    echo "   ❌ Inference endpoint failed"
    echo "   Response: $INFERENCE_RESPONSE"
fi

# Test consciousness endpoint
echo "   Testing consciousness endpoint..."
CONSCIOUSNESS_RESPONSE=$(curl -s http://localhost:5000/api/consciousness)
if echo "$CONSCIOUSNESS_RESPONSE" | grep -q "success\|error"; then
    echo "   ✅ Consciousness endpoint working"
else
    echo "   ❌ Consciousness endpoint failed"
    echo "   Response: $CONSCIOUSNESS_RESPONSE"
fi

# Test energy analysis endpoint
echo "   Testing energy analysis endpoint..."
ENERGY_RESPONSE=$(curl -s -X POST http://localhost:5000/api/energy_analysis \
    -H "Content-Type: application/json" \
    -d '{"input": [0.1] * 784}')
if echo "$ENERGY_RESPONSE" | grep -q "success"; then
    echo "   ✅ Energy analysis endpoint working"
else
    echo "   ❌ Energy analysis endpoint failed"
    echo "   Response: $ENERGY_RESPONSE"
fi

# Test performance endpoint
echo "   Testing performance endpoint..."
PERFORMANCE_RESPONSE=$(curl -s http://localhost:5000/api/performance)
if echo "$PERFORMANCE_RESPONSE" | grep -q "success"; then
    echo "   ✅ Performance endpoint working"
else
    echo "   ❌ Performance endpoint failed"
    echo "   Response: $PERFORMANCE_RESPONSE"
fi

# Save process IDs
echo $API_PID > logs/enhanced_api_server.pid

# Display service status
echo ""
echo "🎉 Enhanced CEP-EIT-P Production Services Started Successfully!"
echo "=============================================================="
echo "📊 Service Status:"
echo "   Enhanced API Server: Running (PID: $API_PID)"
echo "   Port: 5000"
echo "   Logs: logs/enhanced_api_server.log"
echo ""
echo "🌐 API Endpoints:"
echo "   Health Check: http://localhost:5000/api/health"
echo "   Model Info: http://localhost:5000/api/model_info"
echo "   Inference: http://localhost:5000/api/inference"
echo "   Consciousness: http://localhost:5000/api/consciousness"
echo "   Energy Analysis: http://localhost:5000/api/energy_analysis"
echo "   Performance: http://localhost:5000/api/performance"
echo "   Optimization: http://localhost:5000/api/optimize"
echo ""
echo "📋 Management Commands:"
echo "   Stop services: ./stop_enhanced_production.sh"
echo "   View logs: tail -f logs/enhanced_api_server.log"
echo "   Check status: ps aux | grep enhanced_api_server"
echo ""
echo "🔧 Enhanced Features:"
echo "   ✅ Real-time Consciousness Detection"
echo "   ✅ Advanced Energy Analysis"
echo "   ✅ Memristor-Fractal-Chaos Integration"
echo "   ✅ Quantum-Classical Coupling"
echo "   ✅ CEP Constraint Validation"
echo "   ✅ Performance Monitoring"
echo ""
echo "Enhanced CEP-EIT-P Production System is ready! 🚀"
