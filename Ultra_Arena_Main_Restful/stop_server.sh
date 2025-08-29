#!/bin/bash

# Stop Ultra Arena Main RESTful API Server
# This script stops any running server instances
# Automatically detects and uses the appropriate Python version

echo "🛑 Stopping Ultra Arena Main RESTful API Server..."
echo ""

# Function to detect the appropriate Python command
detect_python() {
    local python_cmd=""
    
    # Check OS type
    case "$(uname -s)" in
        Darwin*)    # macOS
            if command -v python3 &> /dev/null; then
                python_cmd="python3"
            elif command -v python &> /dev/null; then
                python_cmd="python"
            fi
            ;;
        Linux*)     # Linux
            if command -v python3 &> /dev/null; then
                python_cmd="python3"
            elif command -v python &> /dev/null; then
                python_cmd="python"
            fi
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)  # Windows
            if command -v python &> /dev/null; then
                python_cmd="python"
            elif command -v python3 &> /dev/null; then
                python_cmd="python3"
            fi
            ;;
        *)          # Unknown OS
            if command -v python3 &> /dev/null; then
                python_cmd="python3"
            elif command -v python &> /dev/null; then
                python_cmd="python"
            fi
            ;;
    esac
    
    echo "$python_cmd"
}

# Detect Python command
PYTHON_CMD=$(detect_python)

# Check if server is running (try both python and python3 patterns)
if pgrep -f "python.*server.py" > /dev/null || pgrep -f "python3.*server.py" > /dev/null; then
    echo "🔄 Found running server. Stopping..."
    
    # Get the process IDs for both patterns
    PIDS=$(pgrep -f "python.*server.py" 2>/dev/null; pgrep -f "python3.*server.py" 2>/dev/null)
    echo "📋 Found server processes: $PIDS"
    
    # Kill the processes (both patterns)
    pkill -f "python.*server.py" 2>/dev/null
    pkill -f "python3.*server.py" 2>/dev/null
    
    # Wait a moment for processes to stop
    sleep 2
    
    # Check if processes are still running
    if pgrep -f "python.*server.py" > /dev/null || pgrep -f "python3.*server.py" > /dev/null; then
        echo "⚠️  Some processes still running. Force killing..."
        pkill -9 -f "python.*server.py" 2>/dev/null
        pkill -9 -f "python3.*server.py" 2>/dev/null
        sleep 1
    fi
    
    # Final check
    if pgrep -f "python.*server.py" > /dev/null || pgrep -f "python3.*server.py" > /dev/null; then
        echo "❌ Failed to stop all server processes."
        echo "📋 Remaining processes:"
        pgrep -f "python.*server.py" 2>/dev/null | xargs ps -p 2>/dev/null
        pgrep -f "python3.*server.py" 2>/dev/null | xargs ps -p 2>/dev/null
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
