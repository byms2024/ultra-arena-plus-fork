#!/bin/bash

# Stop Ultra Arena Main RESTful API Server
# This script stops any running server instances

echo "🛑 Stopping Ultra Arena Main RESTful API Server..."
echo ""

# Check if server is running
if pgrep -f "python.*server.py" > /dev/null; then
    echo "🔄 Found running server. Stopping..."
    
    # Get the process IDs
    PIDS=$(pgrep -f "python.*server.py")
    echo "📋 Found server processes: $PIDS"
    
    # Kill the processes
    pkill -f "python.*server.py"
    
    # Wait a moment for processes to stop
    sleep 2
    
    # Check if processes are still running
    if pgrep -f "python.*server.py" > /dev/null; then
        echo "⚠️  Some processes still running. Force killing..."
        pkill -9 -f "python.*server.py"
        sleep 1
    fi
    
    # Final check
    if pgrep -f "python.*server.py" > /dev/null; then
        echo "❌ Failed to stop all server processes."
        echo "📋 Remaining processes:"
        pgrep -f "python.*server.py" | xargs ps -p
        exit 1
    else
        echo "✅ Server stopped successfully!"
    fi
else
    echo "ℹ️  No server processes found."
fi

# Check if port 5002 is still in use
if lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5002 is still in use. Killing processes on port 5002..."
    lsof -ti:5002 | xargs kill -9
    sleep 1
    
    if lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Failed to free port 5002."
        exit 1
    else
        echo "✅ Port 5002 freed successfully!"
    fi
else
    echo "✅ Port 5002 is free."
fi

echo ""
echo "🎉 Server shutdown complete!"
