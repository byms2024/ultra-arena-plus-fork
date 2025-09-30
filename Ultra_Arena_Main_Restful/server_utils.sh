#!/bin/bash

# Ultra Arena Main RESTful API Server Utilities
# This file contains common functions used by server startup and shutdown scripts

# Determine directory of this utilities script (also the server root)
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Function to check if server is already running
check_server_running() {
    local python_cmd="$1"
    
    if pgrep -f "$python_cmd.*server.py" > /dev/null; then
        echo "⚠️  Server is already running. Stopping existing server..."
        pkill -f "$python_cmd.*server.py"
        sleep 2
        return 0
    fi
    return 1
}

# Function to check and free port 5002
check_and_free_port() {
    if lsof -Pi :5002 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port 5002 is in use. Killing process on port 5002..."
        lsof -ti:5002 | xargs kill -9
        sleep 2
        return 0
    fi
    return 1
}

# Function to detect if running under systemd
is_running_under_systemd() {
    if [ -n "$INVOCATION_ID" ] || [ -n "$JOURNAL_STREAM" ]; then
        return 0
    fi
    return 1
}

# Function to check if systemctl is available
has_systemctl() {
    if command -v systemctl >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Function to start server via systemd
start_server_systemd() {
    local service_name="$1"
    local profile="$2"

    # Export for local process environment (useful if service reads Environment=)
    export RUN_PROFILE="$profile"

    echo "🛠 Using systemd service: ${service_name}.service"
    # Provide env to systemd manager; service may reference it
    systemctl set-environment RUN_PROFILE="$profile" PORT="5002" >/dev/null 2>&1

    # Restart to handle already-running instance cleanly
    if systemctl restart "${service_name}.service"; then
        sleep 2
        if systemctl is-active --quiet "${service_name}.service"; then
            echo "🎉 Service ${service_name}.service is active."
            echo "🔗 Health check: http://localhost:5002/health"
            echo "💡 To see logs: journalctl -u ${service_name}.service -f"
            return 0
        else
            echo "❌ Service did not become active. Inspect logs: journalctl -u ${service_name}.service -xe"
            return 1
        fi
    else
        echo "❌ Failed to start service ${service_name}.service"
        return 1
    fi
}

# Function to start server
start_server() {
    local python_cmd="$1"
    local profile="$2"
    local mode="$3"  # optional: "foreground" to run in foreground
    
    # Set environment variable for the profile
    export RUN_PROFILE="$profile"
    
    # If running under systemd or explicitly requested, run in foreground (no nohup)
    if [ "$mode" = "foreground" ] || is_running_under_systemd; then
        echo "🔄 Starting server in foreground with $python_cmd (systemd-compatible)..."
        echo "🔗 Health check: http://localhost:5002/health"
        exec $python_cmd "$UTILS_DIR/server.py"
        # exec replaces the shell; no return here
    fi
    
    # Default: Start the server in background (for manual runs)
    echo "🔄 Starting server with $python_cmd in background..."
    nohup $python_cmd "$UTILS_DIR/server.py" > server.log 2>&1 &
    local server_pid=$!
    echo "✅ Server started with PID: $server_pid"
    sleep 3
    if pgrep -f "$python_cmd.*server.py" > /dev/null; then
        echo "🎉 Server started successfully!"
        echo "📊 Log file: server.log"
        echo "🔗 Health check: http://localhost:5002/health"
        echo "💡 To stop the server, run: ./stop_server.sh"
        return 0
    else
        echo "❌ Failed to start server. Check server.log for details."
        return 1
    fi
}

# Function to stop server
stop_server() {
    echo "🛑 Stopping Ultra Arena Main RESTful API Server..."
    echo ""
    
    # Prefer stopping via systemd if available
    if has_systemctl; then
        local service_name="${SERVICE_NAME:-ultra-arena-main-restful}"
        echo "🔎 Checking for systemd service: ${service_name}.service"
        if systemctl status "${service_name}.service" >/dev/null 2>&1; then
            echo "🔄 Stopping with systemd..."
            if systemctl stop "${service_name}.service"; then
                systemctl unset-environment RUN_PROFILE PORT >/dev/null 2>&1
                sleep 1
                if systemctl is-active --quiet "${service_name}.service"; then
                    echo "⚠️  Service ${service_name}.service still active after stop."
                else
                    echo "✅ Service ${service_name}.service stopped."
                    echo ""
                    echo "🎉 Server shutdown complete!"
                    return 0
                fi
            else
                echo "⚠️  Failed to stop ${service_name}.service via systemd; falling back to process kill."
            fi
        fi
    fi
    
    # Check if server is running (try both python and python3 patterns)
    if pgrep -f "python.*server.py" > /dev/null || pgrep -f "python3.*server.py" > /dev/null; then
        echo "🔄 Found running server. Stopping..."
        
        # Get the process IDs for both patterns
        local pids=$(pgrep -f "python.*server.py" 2>/dev/null; pgrep -f "python3.*server.py" 2>/dev/null)
        echo "📋 Found server processes: $pids"
        
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
            return 1
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
            return 1
        else
            echo "✅ Port 5002 freed successfully!"
        fi
    else
        echo "✅ Port 5002 is free."
    fi
    
    echo ""
    echo "🎉 Server shutdown complete!"
    return 0
}