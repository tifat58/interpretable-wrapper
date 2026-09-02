#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# Interpretable Wrapper — Start / Stop / Restart
#
# Usage:
#   ./start.sh              Start both services (kill existing first)
#   ./start.sh stop         Stop both services
#   ./start.sh restart      Stop then start
#   ./start.sh --install    Start and install dependencies first
# ═══════════════════════════════════════════════════════════════════════

# ─── Configuration ────────────────────────────────────────────────────
PYENV_ENV="/mnt/sdz/pyenv/versions/3.10.14/envs/cbm-wrapper-env"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
BACKEND_PORT=5000
FRONTEND_PORT=5173
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/.pids"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ─── Kill processes on a port ─────────────────────────────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        warn "Killing process(es) on port $port (PIDs: $(echo $pids | tr '\n' ' '))"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# ─── Stop command ─────────────────────────────────────────────────────
do_stop() {
    info "Stopping services..."
    # Kill by saved PIDs
    if [[ -f "$PID_FILE" ]]; then
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                info "Killed PID $pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    # Also kill anything still on the ports
    kill_port $BACKEND_PORT
    kill_port $FRONTEND_PORT
    ok "Services stopped."
}

# ─── Handle arguments ────────────────────────────────────────────────
ACTION="start"
INSTALL=false
for arg in "$@"; do
    case "$arg" in
        stop)       ACTION="stop" ;;
        restart)    ACTION="restart" ;;
        --install)  INSTALL=true ;;
        -h|--help)
            echo "Usage: $0 [start|stop|restart] [--install]"
            exit 0 ;;
    esac
done

if [[ "$ACTION" == "stop" ]]; then
    do_stop
    exit 0
fi

if [[ "$ACTION" == "restart" ]]; then
    do_stop
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     Interpretable Wrapper — Starting...         ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ─── 1. Activate Python environment ──────────────────────────────────
if [[ -f "$PYENV_ENV/bin/activate" ]]; then
    source "$PYENV_ENV/bin/activate"
    ok "Python: $(python --version 2>&1)"
else
    fail "Python virtualenv not found at $PYENV_ENV"
    exit 1
fi

# ─── 2. Load NVM / Node ──────────────────────────────────────────────
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    source "$NVM_DIR/nvm.sh"
    ok "Node: $(node --version 2>&1)"
else
    # Fallback: check if node is already on PATH
    if command -v node &>/dev/null; then
        ok "Node: $(node --version 2>&1)"
    else
        fail "Node.js not found. Install NVM or add node to PATH."
        exit 1
    fi
fi

# ─── 3. Install dependencies (only with --install) ───────────────────
if [[ "$INSTALL" == true ]]; then
    info "Installing Python dependencies..."
    pip install -q -r "$PROJECT_DIR/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121
    ok "Python dependencies installed."

    info "Installing frontend dependencies..."
    cd "$FRONTEND_DIR" && npm install --silent 2>/dev/null && cd "$PROJECT_DIR"
    ok "Frontend dependencies installed."
fi

# ─── 4. Free ports ───────────────────────────────────────────────────
info "Checking ports $BACKEND_PORT and $FRONTEND_PORT..."
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT
ok "Ports are free."

# ─── 5. Prepare logs ─────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
> "$LOG_DIR/backend.log"
> "$LOG_DIR/frontend.log"

# ─── 6. Start backend ────────────────────────────────────────────────
info "Starting Flask backend (port $BACKEND_PORT)..."
cd "$BACKEND_DIR"
nohup python app.py >> "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_DIR"

# ─── 7. Start frontend ───────────────────────────────────────────────
info "Starting Vite frontend (port $FRONTEND_PORT)..."
cd "$FRONTEND_DIR"
nohup npx vite --host 0.0.0.0 --port $FRONTEND_PORT --strictPort >> "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_DIR"

# Save PIDs for stop command
echo "$BACKEND_PID" > "$PID_FILE"
echo "$FRONTEND_PID" >> "$PID_FILE"

# ─── 8. Wait and verify ──────────────────────────────────────────────
info "Waiting for services to initialize..."

# Wait for backend (up to 30s — model loading can be slow)
BACKEND_OK=false
for i in $(seq 1 30); do
    if curl -sf --max-time 2 "http://127.0.0.1:$BACKEND_PORT/domains" | grep -q '"domains"'; then
        BACKEND_OK=true
        break
    fi
    sleep 1
done

# Wait for frontend (up to 15s)
FRONTEND_OK=false
for i in $(seq 1 15); do
    if curl -sf --max-time 2 "http://127.0.0.1:$FRONTEND_PORT/" | grep -q '<div id="root"'; then
        FRONTEND_OK=true
        break
    fi
    sleep 1
done

# ─── 9. Report ───────────────────────────────────────────────────────
echo ""
NETWORK_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

if $BACKEND_OK; then
    ok "Backend  ✔  http://127.0.0.1:$BACKEND_PORT  (PID $BACKEND_PID)"
else
    fail "Backend  ✘  not responding — check $LOG_DIR/backend.log"
fi

if $FRONTEND_OK; then
    ok "Frontend ✔  http://127.0.0.1:$FRONTEND_PORT  (PID $FRONTEND_PID)"
else
    fail "Frontend ✘  not responding — check $LOG_DIR/frontend.log"
fi

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
if $BACKEND_OK && $FRONTEND_OK; then
    echo -e "${BOLD}║  ${GREEN}All services are running!${NC}${BOLD}                       ║${NC}"
else
    echo -e "${BOLD}║  ${RED}Some services failed to start${NC}${BOLD}                   ║${NC}"
fi
echo -e "${BOLD}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  Local:   http://localhost:$FRONTEND_PORT               ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Network: http://$NETWORK_IP:$FRONTEND_PORT       ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Logs:    $LOG_DIR/           ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Stop:    ${CYAN}./start.sh stop${NC}                           ${BOLD}║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
