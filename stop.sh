#!/usr/bin/env bash
set -uo pipefail

# ─── Configuration ────────────────────────────────────────────────────
BACKEND_PORT=5000
FRONTEND_PORT=5173
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ─── 1. Stop by saved PID files (if present) ─────────────────────────
stop_pidfile() {
    local name=$1 file=$2
    if [ -f "$file" ]; then
        local pid
        pid=$(cat "$file" 2>/dev/null || true)
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            warn "Stopping $name (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$file"
    fi
}

stop_pidfile "backend"  "$LOG_DIR/backend.pid"
stop_pidfile "frontend" "$LOG_DIR/frontend.pid"

# ─── 2. Sweep the ports (catches anything PID files missed) ──────────
kill_port() {
    local port=$1 pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        warn "Killing process(es) on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    else
        ok "Port $port already free."
    fi
}

info "Freeing ports $BACKEND_PORT and $FRONTEND_PORT..."
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT

echo ""
ok "All demo services stopped."
