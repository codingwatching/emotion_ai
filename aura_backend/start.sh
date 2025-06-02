#!/bin/bash
# Aura Clean Start Script
# Simplified startup that properly handles the architecture

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[Aura]${NC} $1"; }
error() { echo -e "${RED}[Error]${NC} $1" >&2; }

# Cleanup on exit
cleanup() {
    log "Stopping services..."
    pkill -P $$ 2>/dev/null || true
    wait
    log "Services stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Check environment
if [ ! -d ".venv" ]; then
    error "Virtual environment not found!"
    error "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate venv
# shellcheck source=.venv/bin/activate
source .venv/bin/activate

# Start API (which includes Aura internal tools)
log "Starting Aura API Server..."
python -m main &
export API_PID=$!

# Wait for API
log "Waiting for API to start..."
for _ in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        log "API is ready!"
        break
    fi
    sleep 1
done

# Optional: Start frontend
if [[ "$1" == "--with-frontend" ]]; then
    log "Starting frontend..."
    (cd .. && npm run dev) &
    sleep 3
    log "Frontend started!"
fi

# Show status
echo ""
log "✅ Aura is running!"
echo "  📡 API: http://localhost:8000"
echo "  📖 Docs: http://localhost:8000/docs"
[[ "$1" == "--with-frontend" ]] && echo "  🌐 UI: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"

# Keep running
wait
