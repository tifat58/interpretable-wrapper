#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────
# This MVP runs on the local .venv (Flask is installed there), NOT a conda env.
VENV_DIR=".venv"
BACKEND_PORT=5000
FRONTEND_PORT=5173
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_PY="$PROJECT_DIR/$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ─── 1. Locate the Python virtual environment ────────────────────────
info "Locating Python environment..."
if [ ! -x "$VENV_PY" ]; then
    info "Virtual environment not found at $VENV_DIR — creating it..."
    python3 -m venv "$PROJECT_DIR/$VENV_DIR"
    ok "Created virtual environment at $VENV_DIR"
fi
ok "Using Python: $("$VENV_PY" --version) at $VENV_PY"

# ─── 2. Install Python dependencies ──────────────────────────────────
info "Installing Python dependencies..."
"$VENV_PY" -m pip install -q -r "$PROJECT_DIR/requirements.txt"
ok "Python dependencies installed."

# ─── 4. Install frontend dependencies ────────────────────────────────
info "Installing frontend dependencies..."
cd "$FRONTEND_DIR"
npm install --silent 2>/dev/null
ok "Frontend dependencies installed."
cd "$PROJECT_DIR"

# ─── 5. Kill anything on our ports ───────────────────────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        warn "Killing process(es) on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

info "Freeing ports $BACKEND_PORT and $FRONTEND_PORT..."
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT
ok "Ports are free."

# ─── 6. Prepare log directory ────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ─── 7. Start backend ────────────────────────────────────────────────
info "Starting Flask backend on port $BACKEND_PORT..."
cd "$BACKEND_DIR"
nohup "$VENV_PY" app.py > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_DIR"
echo "$BACKEND_PID" > "$LOG_DIR/backend.pid"
info "Backend PID: $BACKEND_PID"

# ─── 8. Start frontend ───────────────────────────────────────────────
info "Starting Vite frontend on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
nohup npx vite --port $FRONTEND_PORT --strictPort > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_DIR"
echo "$FRONTEND_PID" > "$LOG_DIR/frontend.pid"
info "Frontend PID: $FRONTEND_PID"

# ─── 9. Wait and verify ──────────────────────────────────────────────
info "Waiting for services to start..."
sleep 4

PASS=true

# Check backend
if curl -s --max-time 5 -X POST "http://127.0.0.1:$BACKEND_PORT/predict" \
    -H "Content-Type: application/json" \
    -d '{"input_type":"text","data":"test"}' | grep -q '"label"'; then
    ok "Backend is running  → http://127.0.0.1:$BACKEND_PORT"
else
    fail "Backend is NOT responding on port $BACKEND_PORT"
    fail "Check logs: $LOG_DIR/backend.log"
    PASS=false
fi

# Check frontend
if curl -s --max-time 5 "http://localhost:$FRONTEND_PORT/" | grep -q '<div id="root"'; then
    ok "Frontend is running → http://localhost:$FRONTEND_PORT"
else
    fail "Frontend is NOT responding on port $FRONTEND_PORT"
    fail "Check logs: $LOG_DIR/frontend.log"
    PASS=false
fi

# Check proxy (frontend → backend)
if curl -s --max-time 5 -X POST "http://localhost:$FRONTEND_PORT/predict" \
    -H "Content-Type: application/json" \
    -d '{"input_type":"text","data":"proxy test"}' | grep -q '"label"'; then
    ok "Proxy working (frontend → backend)"
else
    warn "Proxy not responding — API calls from the browser may fail"
    PASS=false
fi

# ─── 10. Summary ─────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────"
if [ "$PASS" = true ]; then
    ok "All services are up and running!"
else
    fail "Some services failed to start. Check logs in: $LOG_DIR/"
fi
echo ""
info "Backend:  http://127.0.0.1:$BACKEND_PORT  (PID $BACKEND_PID)"
info "Frontend: http://localhost:$FRONTEND_PORT  (PID $FRONTEND_PID)"
info "Logs:     $LOG_DIR/"
echo ""
info "To stop:  ./stop.sh"
echo "─────────────────────────────────────────────────"
