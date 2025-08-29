#!/bin/bash

# Start Ultra Arena Main RESTful API Server with BR Profile
# This script starts the server using the br_profile_restful configuration
# Automatically detects and uses the appropriate Python version

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_utils.sh"

echo "🚀 Starting Ultra Arena Main RESTful API Server with BR Profile..."
echo "📋 Profile: br_profile_restful"
echo "🌐 Port: 5002"
echo ""

# Detect Python command
PYTHON_CMD=$(detect_python)
if [ $? -ne 0 ] || [ -z "$PYTHON_CMD" ]; then
    echo "❌ Failed to detect Python installation"
    echo "💡 Please install Python 3.x and try again"
    exit 1
fi

echo ""

# Check if server is already running
check_server_running "$PYTHON_CMD"

# Check if port 5002 is in use
check_and_free_port

# Start the server
if start_server "$PYTHON_CMD" "br_profile_restful"; then
    exit 0
else
    exit 1
fi
