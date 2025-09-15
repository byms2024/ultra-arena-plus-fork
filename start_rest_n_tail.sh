#!/bin/sh

# Save the current directory
ORIGINAL_DIR="$(pwd)"

# Get the absolute path to the start_server_br.sh script
SCRIPT_PATH="./Ultra_Arena_Main_Restful/start_server_br.sh"

# Change to the script's directory before running it
cd "$(dirname "$SCRIPT_PATH")"

# Execute the script
./start_server_br.sh

tail -f ./server.log

# Return to the original directory
cd "$ORIGINAL_DIR"
