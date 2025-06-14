#!/bin/bash
echo "🌟 Starting Aura Complete Backend System..."

# Function to cleanup background processes
cleanup() {
    echo "Stopping all Aura services..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start API server in background
echo "🚀 Starting API Server..."
./start_api.sh &
API_PID=$!

# Wait a moment for API server to start
sleep 3

# Start MCP server in background
echo "🔗 Starting MCP Server..."
./start_mcp.sh &
MCP_PID=$!

echo "✅ All services started!"
echo "📡 API Server: http://localhost:8000"
echo "🔗 MCP Server: Available for external connections"
echo "📖 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
