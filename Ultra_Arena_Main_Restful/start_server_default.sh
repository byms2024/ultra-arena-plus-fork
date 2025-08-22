#!/bin/bash

# Start Ultra Arena Main RESTful API Server with Default Profile
# This script starts the server using the default_profile_restful configuration

echo "🚀 Starting Ultra Arena Main RESTful API Server with Default Profile..."
echo "📋 Profile: default_profile_restful"
echo "🌐 Port: 5002"
echo ""

# Check if server is already running
if pgrep -f "python.*server.py" > /dev/null; then
    echo "⚠️  Server is already running. Stopping existing server..."
    pkill -f "python.*server.py"
    sleep 2
fi

# Check if port 5002 is in use
if lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5002 is in use. Killing process on port 5002..."
    lsof -ti:5002 | xargs kill -9
    sleep 2
fi

# Set environment variable for the profile
export RUN_PROFILE=default_profile_restful

# Start the server in background
echo "🔄 Starting server..."
nohup python server.py > server.log 2>&1 &

# Get the process ID
SERVER_PID=$!
echo "✅ Server started with PID: $SERVER_PID"

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if pgrep -f "python.*server.py" > /dev/null; then
    echo "🎉 Server started successfully!"
    echo "📊 Log file: server.log"
    echo "🔗 Health check: http://localhost:5002/health"
    echo "📋 API endpoints:"
    echo "   - GET  /health - Health check"
    echo "   - POST /api/process/combo - Process combo (synchronous)"
    echo "   - POST /api/process/combo/async - Process combo (asynchronous)"
    echo "   - GET  /api/requests/<request_id> - Get request status"
    echo "   - GET  /api/requests - Get all requests"
    echo "   - GET  /api/combos - Get available combos"
    echo ""
    echo "💡 To stop the server, run: ./stop_server.sh"
else
    echo "❌ Failed to start server. Check server.log for details."
    exit 1
fi
