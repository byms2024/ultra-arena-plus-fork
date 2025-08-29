#!/bin/bash

# Stop Ultra Arena Main RESTful API Server
# This script stops any running server instances
# Automatically detects and uses the appropriate Python version

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_utils.sh"

# Stop the server
if stop_server; then
    exit 0
else
    exit 1
fi
