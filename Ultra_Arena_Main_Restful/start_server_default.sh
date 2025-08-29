#!/bin/bash

# Start Ultra Arena Main RESTful API Server with Default Profile
# This script starts the server using the default_profile_restful configuration
# Automatically detects and uses the appropriate Python version

echo "🚀 Starting Ultra Arena Main RESTful API Server with Default Profile..."
echo "📋 Profile: default_profile_restful"
echo "🌐 Port: 5002"
echo ""

# Function to detect the appropriate Python command
detect_python() {
    local python_cmd=""
    
    # Check OS type
    case "$(uname -s)" in
        Darwin*)    # macOS
            echo "🍎 Detected macOS" >&2
            # Try python3 first (most common on macOS)
            if command -v python3 &> /dev/null; then
                python_cmd="python3"
                echo "✅ Using python3" >&2
            elif command -v python &> /dev/null; then
                python_cmd="python"
                echo "✅ Using python" >&2
            else
                echo "❌ No Python installation found" >&2
                return 1
            fi
            ;;
        Linux*)     # Linux
            echo "🐧 Detected Linux" >&2
            # Try python3 first (most common on modern Linux)
            if command -v python3 &> /dev/null; then
                python_cmd="python3"
                echo "✅ Using python3" >&2
            elif command -v python &> /dev/null; then
                python_cmd="python"
                echo "✅ Using python" >&2
            else
                echo "❌ No Python installation found" >&2
                return 1
            fi
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)  # Windows
            echo "🪟 Detected Windows" >&2
            # Try python first (most common on Windows)
            if command -v python &> /dev/null; then
                python_cmd="python"
                echo "✅ Using python" >&2
            elif command -v python3 &> /dev/null; then
                python_cmd="python3"
                echo "✅ Using python3" >&2
            else
                echo "❌ No Python installation found" >&2
                return 1
            fi
            ;;
        *)          # Unknown OS
            echo "❓ Unknown operating system: $(uname -s)" >&2
            # Try both python and python3
            if command -v python3 &> /dev/null; then
                python_cmd="python3"
                echo "✅ Using python3" >&2
            elif command -v python &> /dev/null; then
                python_cmd="python"
                echo "✅ Using python" >&2
            else
                echo "❌ No Python installation found" >&2
                return 1
            fi
            ;;
    esac
    
    # Verify Python version
    if [ -n "$python_cmd" ]; then
        local version=$($python_cmd --version 2>&1)
        echo "📋 Python version: $version" >&2
        
        # Check if it's Python 3.x
        if [[ $version == Python\ 3.* ]]; then
            echo "✅ Python 3.x detected - compatible" >&2
        else
            echo "⚠️  Warning: Non-Python 3.x version detected" >&2
            echo "   This might cause compatibility issues" >&2
        fi
    fi
    
    echo "$python_cmd"
}

# Detect Python command
PYTHON_CMD=$(detect_python)
if [ $? -ne 0 ] || [ -z "$PYTHON_CMD" ]; then
    echo "❌ Failed to detect Python installation"
    echo "💡 Please install Python 3.x and try again"
    exit 1
fi

echo ""

# Check if server is already running
if pgrep -f "$PYTHON_CMD.*server.py" > /dev/null; then
    echo "⚠️  Server is already running. Stopping existing server..."
    pkill -f "$PYTHON_CMD.*server.py"
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
echo "🔄 Starting server with $PYTHON_CMD..."
nohup $PYTHON_CMD server.py > server.log 2>&1 &

# Get the process ID
SERVER_PID=$!
echo "✅ Server started with PID: $SERVER_PID"

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if pgrep -f "$PYTHON_CMD.*server.py" > /dev/null; then
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
