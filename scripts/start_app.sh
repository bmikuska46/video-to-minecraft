#!/usr/bin/env bash
# Start the local app: the Docker Compose backend (Postgres, Redis, MinIO, API,
# GPU pipeline worker) plus the Expo dev server for the phone app.
#
# Press Ctrl+C once to stop everything safely:
#   1. the Expo dev server exits (it receives the Ctrl+C directly),
#   2. the API stops, so no new jobs are queued,
#   3. the worker finishes the job it is running (Celery warm shutdown),
#   4. MinIO, Redis and Postgres shut down cleanly (data stays in Docker volumes).
# Press Ctrl+C again while a job is still running to abandon that job and stop now.
#
# Usage: scripts/start_app.sh [--backend-only] [--no-build]
# Environment overrides:
#   VTM_LAN_IP               address the phone uses to reach this machine (auto-detected)
#   EXPO_PUBLIC_API_URL      API URL baked into the Expo bundle (default http://$VTM_LAN_IP:8000)
#   VTM_S3_PUBLIC_ENDPOINT_URL  MinIO URL used in signed links (default http://$VTM_LAN_IP:9000)
#   VTM_WORKER_STOP_TIMEOUT  seconds to wait for a running job on Ctrl+C (default 1800)

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$ROOT/infra/compose.yaml")
LOG_DIR="$ROOT/artifacts/logs"
WORKER_STOP_TIMEOUT="${VTM_WORKER_STOP_TIMEOUT:-1800}"
READY_TIMEOUT=300

BACKEND_ONLY=0
BUILD=1
for argument in "$@"; do
  case "$argument" in
    --backend-only) BACKEND_ONLY=1 ;;
    --no-build) BUILD=0 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'Unknown option: %s (see --help)\n' "$argument" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m[app]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[app]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[app]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- shutdown ---

BACKEND_STARTED=0
STOPPING=0
FORCE=0
LOGS_PID=""

on_interrupt_while_stopping() {
  if (( ! FORCE )); then
    FORCE=1
    warn "Second Ctrl+C: not waiting for the running job any longer."
  fi
}

worker_active_jobs() {
  # Prints the number of tasks the worker is executing, or "unknown".
  local output
  output="$("${COMPOSE[@]}" exec -T pipeline-worker \
    celery -A app.pipeline:celery_app inspect active --json --timeout 5 2>/dev/null)" || {
    echo unknown; return
  }
  python3 -c '
import json, sys
text = sys.stdin.read()
start = text.find("{")
try:
    replies = json.loads(text[start:]) if start >= 0 else {}
    print(sum(len(tasks or []) for tasks in replies.values()))
except ValueError:
    print("unknown")
' <<<"$output"
}

stop_service() {
  local service="$1" timeout="$2"
  "${COMPOSE[@]}" stop --timeout "$timeout" "$service" >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  (( STOPPING )) && return
  STOPPING=1
  set +e
  trap on_interrupt_while_stopping INT TERM
  echo
  say "Stopping the app safely..."

  if (( BACKEND_STARTED )); then
    say "Stopping the API (no new jobs)..."
    stop_service api 20

    local active
    active="$(worker_active_jobs)"
    if [[ "$active" == "0" ]]; then
      say "Worker is idle; stopping it..."
      stop_service pipeline-worker 30
    else
      if [[ "$active" == "unknown" ]]; then
        warn "Could not ask the worker whether it is busy; treating it as busy."
      else
        warn "Worker is running $active job(s). Waiting up to ${WORKER_STOP_TIMEOUT}s for it to finish."
      fi
      warn "Press Ctrl+C again to stop now (the interrupted scan/export must be resubmitted)."
      # Celery finishes the current task on SIGTERM; poll so a second Ctrl+C can cut the wait short.
      "${COMPOSE[@]}" kill --signal SIGTERM pipeline-worker >/dev/null 2>&1
      local waited=0
      while (( ! FORCE && waited < WORKER_STOP_TIMEOUT )) \
        && [[ "$("${COMPOSE[@]}" ps --status running --quiet pipeline-worker 2>/dev/null)" ]]; do
        sleep 2
        waited=$((waited + 2))
      done
      if [[ "$("${COMPOSE[@]}" ps --status running --quiet pipeline-worker 2>/dev/null)" ]]; then
        warn "Worker did not finish in time; stopping it now."
      fi
      stop_service pipeline-worker 10
    fi

    say "Stopping MinIO, Redis and Postgres..."
    "${COMPOSE[@]}" stop --timeout 30 >/dev/null 2>&1
  fi

  if [[ -n "$LOGS_PID" ]]; then
    kill "$LOGS_PID" 2>/dev/null
    wait "$LOGS_PID" 2>/dev/null
  fi

  if (( BACKEND_STARTED )); then
    local still_running
    still_running="$("${COMPOSE[@]}" ps --status running --services 2>/dev/null)"
    if [[ -n "$still_running" ]]; then
      warn "Still running: $(tr '\n' ' ' <<<"$still_running")"
      status=1
    else
      say "All services stopped. Data is kept in Docker volumes."
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
# Ctrl+C before Expo starts should still stop what was started.
trap 'exit 130' INT TERM

# --------------------------------------------------------------- preflight ---

command -v docker >/dev/null || fail "Docker is not installed."
docker info >/dev/null 2>&1 || fail "Cannot talk to the Docker daemon (is it running, and are you in the docker group?)."
docker compose version >/dev/null 2>&1 || fail "The Docker Compose plugin is missing."
[[ -f "$ROOT/infra/paper/paper-26.2-build.112-stable.jar" || -n "${VTM_PAPER_JAR_PATH:-}" ]] \
  || fail "Paper server JAR is missing: infra/paper/paper-26.2-build.112-stable.jar (or set VTM_PAPER_JAR_PATH)."

if (( ! BACKEND_ONLY )); then
  command -v pnpm >/dev/null || fail "pnpm is not installed (needed for the Expo dev server); or use --backend-only."
  [[ -d "$ROOT/apps/mobile/node_modules" ]] \
    || fail "Mobile dependencies are missing. Run: pnpm --dir apps/mobile install"
  if ss -ltnH 'sport = :8081' 2>/dev/null | grep -q .; then
    fail "Port 8081 is already in use; another Expo/Metro server is probably running."
  fi
fi

if [[ -z "${VTM_LAN_IP:-}" ]]; then
  VTM_LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null \
    | awk '{for (i = 1; i < NF; i++) if ($i == "src") { print $(i + 1); exit }}')"
  VTM_LAN_IP="${VTM_LAN_IP:-127.0.0.1}"
fi
export VTM_S3_PUBLIC_ENDPOINT_URL="${VTM_S3_PUBLIC_ENDPOINT_URL:-http://$VTM_LAN_IP:9000}"
export EXPO_PUBLIC_API_URL="${EXPO_PUBLIC_API_URL:-http://$VTM_LAN_IP:8000}"

# ------------------------------------------------------------------- start ---

mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend-$(date +%Y%m%d-%H%M%S).log"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if (( BUILD )); then
  say "Building images (cached layers make this quick when nothing changed)..."
  "${COMPOSE[@]}" build --quiet || fail "Image build failed."
fi

say "Starting backend services..."
BACKEND_STARTED=1
if ! up_output="$("${COMPOSE[@]}" up --detach --no-build --remove-orphans 2>&1)"; then
  printf '%s\n' "$up_output" >&2
  fail "docker compose up failed."
fi
"${COMPOSE[@]}" logs --follow --no-color --since "$STARTED_AT" >"$BACKEND_LOG" 2>&1 &
LOGS_PID=$!

say "Waiting for the API and the GPU worker to become ready (up to ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
api_ready=0
worker_ready=0
while (( SECONDS < deadline )); do
  if (( ! api_ready )) && curl -fs -o /dev/null http://127.0.0.1:8000/health/ready; then
    api_ready=1
    say "API ready: http://127.0.0.1:8000 (phone: $EXPO_PUBLIC_API_URL)"
  fi
  if (( ! worker_ready )); then
    worker_container="$("${COMPOSE[@]}" ps --all --quiet pipeline-worker)"
    worker_health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
      "$worker_container" 2>/dev/null || echo missing)"
    case "$worker_health" in
      healthy) worker_ready=1; say "GPU pipeline worker ready." ;;
      unhealthy|exited|dead|missing)
        tail -n 30 "$BACKEND_LOG" >&2
        fail "Pipeline worker failed to start ($worker_health); full log: $BACKEND_LOG" ;;
    esac
  fi
  (( api_ready && worker_ready )) && break
  sleep 2
done
(( api_ready && worker_ready )) || {
  tail -n 30 "$BACKEND_LOG" >&2
  fail "Backend did not become ready within ${READY_TIMEOUT}s; full log: $BACKEND_LOG"
}

say "Backend logs: $BACKEND_LOG"
say "MinIO console: http://127.0.0.1:9001 (minioadmin / minioadmin)"

if (( BACKEND_ONLY )); then
  say "Backend is running. Press Ctrl+C to stop it."
  # Blocks until Ctrl+C; the EXIT trap then stops everything.
  wait "$LOGS_PID"
else
  say "Starting the Expo dev server (phone API URL: $EXPO_PUBLIC_API_URL). Press Ctrl+C to stop the app."
  # Expo runs in the foreground so its QR code and key shortcuts work. Ctrl+C reaches
  # Expo and this script together; the INT trap then runs the EXIT cleanup.
  pnpm --dir "$ROOT/apps/mobile" start || true
fi
