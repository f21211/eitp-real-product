#!/bin/bash
"""
Enhanced CEP-EIT-P Production Stop Script
Stop all enhanced production services
"""

echo "🛑 Stopping Enhanced CEP-EIT-P Production Services"
echo "================================================="

# Stop Enhanced API Server
if [ -f logs/enhanced_api_server.pid ]; then
    API_PID=$(cat logs/enhanced_api_server.pid)
    if ps -p $API_PID > /dev/null; then
        echo "🛑 Stopping Enhanced API Server (PID: $API_PID)..."
        kill $API_PID
        sleep 2
        
        # Check if process is still running
        if ps -p $API_PID > /dev/null; then
            echo "⚠️  Process still running, force killing..."
            kill -9 $API_PID
            sleep 1
        fi
        
        if ps -p $API_PID > /dev/null; then
            echo "❌ Failed to stop Enhanced API Server"
        else
            echo "✅ Enhanced API Server stopped successfully"
        fi
    else
        echo "ℹ️  Enhanced API Server not running"
    fi
    rm -f logs/enhanced_api_server.pid
else
    echo "ℹ️  No PID file found for Enhanced API Server"
fi

# Kill any remaining processes
echo "🧹 Cleaning up remaining processes..."
pkill -f enhanced_api_server.py
pkill -f "python3.*enhanced_api_server"

# Wait a moment for processes to stop
sleep 2

# Check for any remaining processes
REMAINING_PROCESSES=$(ps aux | grep enhanced_api_server | grep -v grep | wc -l)
if [ $REMAINING_PROCESSES -gt 0 ]; then
    echo "⚠️  Warning: $REMAINING_PROCESSES enhanced API server processes still running"
    echo "   You may need to manually kill them:"
    ps aux | grep enhanced_api_server | grep -v grep
else
    echo "✅ All enhanced API server processes stopped"
fi

# Display final status
echo ""
echo "📊 Final Status:"
echo "   Enhanced API Server: Stopped"
echo "   Port 5000: Available"
echo ""
echo "🎉 Enhanced CEP-EIT-P Production Services Stopped Successfully!"
echo "=============================================================="
