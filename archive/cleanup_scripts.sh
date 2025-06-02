#!/bin/bash
# Clean up redundant startup scripts

echo "Cleaning up redundant startup scripts..."

# List of scripts to remove
SCRIPTS_TO_REMOVE=(
    "start_all.sh"
    "start_complete.sh" 
    "start_mcp.sh"
    "start_mcp_background.sh"
    "start_mcp_services.sh"
    "start_unified.sh"
    "start_simple.sh"
    "very-bad-do-not-use-this-start_backend.py"
)

# Remove each script
for script in "${SCRIPTS_TO_REMOVE[@]}"; do
    if [ -f "$script" ]; then
        echo "Removing $script..."
        rm "$script"
    fi
done

# Remove any leftover PID files
rm -f .mcp_server.pid .api_server.pid

# Clean up any leftover log files
rm -f aura_server.log aura_mcp_server.log mcp_server.log firebase-debug.log

echo "✅ Cleanup complete!"
echo ""
echo "Remaining startup scripts:"
echo "  - start.sh         (main entry point)"
echo "  - start_api.sh     (API only, for development)"
echo "  - start_frontend.sh (frontend only)"
echo ""
echo "To start Aura:"
echo "  ./start.sh                # Backend only"
echo "  ./start.sh --with-frontend # Backend + Frontend"
