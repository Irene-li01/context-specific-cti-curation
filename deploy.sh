#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

COMPOSE="${COMPOSE:-docker compose}"
ENV_FILE="${ENV_FILE:-.env}"
BRANCH="${BRANCH:-}"
API_PORT="${API_PORT:-8000}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${API_PORT}/api/v1/health}"
SKIP_PULL="${SKIP_PULL:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
PIPELINE_ARGS="${PIPELINE_ARGS:-}"
ALLOW_DIRTY="${DEPLOY_ALLOW_DIRTY:-0}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-1}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
PRESERVE_GENERATED="${PRESERVE_GENERATED:-1}"
GENERATED_STIX_FILE="cti_microservice_v1/curated_threats_stix3.json"
PRESERVED_STIX_COPY=""

log() {
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

fail() {
  printf '\nDeploy failed: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
CCTI deployment script

Run from the server-side repository directory:

  ./deploy.sh

Environment overrides:

  BRANCH=main              Checkout/pull a specific branch before deployment
  SKIP_PULL=1             Do not run git pull
  SKIP_BUILD=1            Do not rebuild Docker image
  SKIP_PIPELINE=1         Do not run the pipeline job
  PIPELINE_ARGS="..."     Optional arguments passed to pipeline.py, e.g. "--use-secbert"
  API_PORT=8000           Port used for local health check
  HEALTH_URL=http://...   Override health check URL
  DEPLOY_ALLOW_DIRTY=1    Allow deployment with local uncommitted changes
  BACKUP_BEFORE_DEPLOY=0  Skip .env and generated-data backup
  BACKUP_DIR=backups      Directory for deployment backups
  PRESERVE_GENERATED=0    Do not auto-preserve generated STIX before git pull

USAGE
}

backup_runtime_files() {
  local timestamp backup_path archive_path
  timestamp="$(date -u '+%Y%m%d-%H%M%S')"
  backup_path="${BACKUP_DIR}/deploy-${timestamp}"
  archive_path="${backup_path}/generated-data.tar.gz"

  mkdir -p "$backup_path"
  cp "$ENV_FILE" "${backup_path}/.env"

  local generated_paths=()
  for path in \
    "data/cleaned" \
    "data/recommendations" \
    "ner" \
    "$GENERATED_STIX_FILE" \
    "cti_microservice_v1/vector_store"
  do
    if [[ -e "$path" ]]; then
      generated_paths+=("$path")
    fi
  done

  if (( ${#generated_paths[@]} > 0 )); then
    tar -czf "$archive_path" "${generated_paths[@]}"
    log "Backed up runtime files to $backup_path"
  else
    log "No generated runtime files found to back up"
  fi
}

preserve_generated_stix_if_needed() {
  if [[ "$PRESERVE_GENERATED" != "1" || ! -f "$GENERATED_STIX_FILE" ]]; then
    return
  fi

  if git diff --quiet -- "$GENERATED_STIX_FILE"; then
    return
  fi

  local changed_files
  changed_files="$(git diff --name-only)"
  if [[ "$changed_files" != "$GENERATED_STIX_FILE" ]]; then
    fail "working tree has uncommitted changes outside $GENERATED_STIX_FILE. Commit/stash them or set DEPLOY_ALLOW_DIRTY=1."
  fi

  PRESERVED_STIX_COPY="/tmp/ccti-curated-threats-stix-$(date -u '+%Y%m%d-%H%M%S').json"
  cp "$GENERATED_STIX_FILE" "$PRESERVED_STIX_COPY"
  git restore "$GENERATED_STIX_FILE"
  log "Temporarily preserved generated STIX file before git pull"
}

restore_preserved_stix_if_needed() {
  if [[ -n "$PRESERVED_STIX_COPY" && -f "$PRESERVED_STIX_COPY" ]]; then
    cp "$PRESERVED_STIX_COPY" "$GENERATED_STIX_FILE"
    log "Restored preserved generated STIX file"
    PRESERVED_STIX_COPY=""
  fi
}

trap 'status=$?; if [[ $status -ne 0 ]]; then restore_preserved_stix_if_needed; fi' EXIT

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

command -v docker >/dev/null 2>&1 || fail "docker is not installed or not in PATH."
$COMPOSE version >/dev/null 2>&1 || fail "'$COMPOSE' is unavailable. Install Docker Compose v2."

[[ -f "docker-compose.yml" ]] || fail "docker-compose.yml not found. Run this script from the CCTI repo root."
[[ -f "$ENV_FILE" ]] || fail "$ENV_FILE not found. Copy .env.example to .env and set production values."

if grep -q "replace-with-your-misp-api-key" "$ENV_FILE"; then
  fail "$ENV_FILE still contains the placeholder MISP_API_KEY."
fi

if [[ "$BACKUP_BEFORE_DEPLOY" == "1" ]]; then
  log "Backing up .env and generated runtime files"
  backup_runtime_files
else
  log "Skipping runtime backup"
fi

if [[ "$SKIP_PULL" != "1" && -d ".git" ]]; then
  if [[ "$ALLOW_DIRTY" != "1" ]]; then
    git diff --cached --quiet || fail "index has staged changes. Commit/stash them or set DEPLOY_ALLOW_DIRTY=1."
    preserve_generated_stix_if_needed
    git diff --quiet || fail "working tree has uncommitted changes. Commit/stash them or set DEPLOY_ALLOW_DIRTY=1."
  fi

  if [[ -n "$BRANCH" ]]; then
    log "Checking out branch: $BRANCH"
    git checkout "$BRANCH"
  fi

  log "Pulling latest code"
  git pull --ff-only
  restore_preserved_stix_if_needed
else
  log "Skipping git pull"
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  log "Building Docker image"
  $COMPOSE build
else
  log "Skipping Docker build"
fi

if [[ "$SKIP_PIPELINE" != "1" ]]; then
  log "Running CTI pipeline: python pipeline.py ${PIPELINE_ARGS}"
  $COMPOSE --profile jobs run --rm pipeline python pipeline.py $PIPELINE_ARGS
else
  log "Skipping pipeline run"
fi

log "Starting API service"
$COMPOSE up -d api

log "Waiting for API health check: $HEALTH_URL"
for attempt in {1..30}; do
  if curl -fsS "$HEALTH_URL" >/dev/null; then
    log "Deployment complete. API is healthy."
    $COMPOSE ps
    exit 0
  fi
  sleep 2
done

log "API logs"
$COMPOSE logs --tail=80 api || true
fail "API health check did not pass after 60 seconds."
