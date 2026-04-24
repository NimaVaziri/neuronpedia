#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="gemma-2-2b"
TL_MODEL_ID="google/gemma-2-2b"
SOURCE_SET="gemmascope-transcoder-16k"
MODEL_SOURCESET="${MODEL_ID}.${SOURCE_SET}"
TRANSCODER_SET="gemma"

WEB_URL="${WEB_URL:-http://localhost:3000}"
INFERENCE_URL="${INFERENCE_URL:-http://localhost:5002}"
GRAPH_URL="${GRAPH_URL:-http://localhost:5004}"

ROOT_ENV=".env"
LOCALHOST_ENV=".env.localhost"
GRAPH_ENV="apps/graph/.env"
WEBAPP_ENV="apps/webapp/.env.localhost"
INFERENCE_ENV=".env.inference.${MODEL_SOURCESET}"

GRAPH_SECRET="${GRAPH_SECRET:-localhost-secret}"
INFERENCE_SECRET="${INFERENCE_SECRET:-localhost-secret}"
LAYERS_SPEC="${LAYERS:-0-25}"
EXPLANATIONS="${EXPLANATIONS:-true}"
IMPORT_JOBS="${IMPORT_JOBS:-1}"
USE_LOCAL_HF_CACHE=1
USE_CUDA=0
SKIP_BUILD=0
START_INFERENCE=0
START_GRAPH=1
GRAPH_MODE="docker"
WEBAPP_MODE="local"

LOG_DIR="${LOG_DIR:-/tmp/neuronpedia-gemma-local}"
GRAPH_MPS_PID_FILE="${LOG_DIR}/graph-mps.pid"
GRAPH_MPS_LOG="${LOG_DIR}/graph-mps.log"
GRAPH_PYTHON="${GRAPH_PYTHON:-apps/graph/.venv/bin/python}"
WEBAPP_PID_FILE="${LOG_DIR}/webapp.pid"
WEBAPP_LOG="${LOG_DIR}/webapp.log"
WEBAPP_RUNTIME_ENV="${LOG_DIR}/webapp.env.localhost"
WEBAPP_RUNTIME_ENV_ABS=""
ALL_STARTED_AT=0

usage() {
  cat <<'EOF'
Usage:
  scripts/gemma-2-2b-local.sh <command> [options]

Commands:
  up       Start local Postgres, webapp, and graph server.
  import   Import Gemma-2-2B Gemmascope transcoder data into local Postgres.
  verify   Print DB counts for imported Gemma-2-2B data.
  smoke    Check webapp, inference, and graph health endpoints.
  all      Run up, smoke, import, verify.
  down     Stop Docker services started by this script and any local webapp/MPS graph process.

Options:
  --layers <spec>          Layers to import. Examples: 0-25, 20, 0,1,2. Default: 0-25.
  --import-jobs <n>        Number of source/layer imports to run concurrently. Default: 1.
  --explanations <mode>    Import explanations mode: true, false, only. Default: true.
  --cuda                   Use CUDA Docker compose overrides for inference and graph.
  --webapp <mode>          Webapp mode: local, docker, skip. Default: local.
  --graph <mode>           Graph mode: docker, mps, skip. Default: docker.
  --with-inference         Also start and smoke-check the standalone inference server. Default: off.
  --no-inference           Compatibility no-op; standalone inference is already off by default.
  --no-graph               Do not start or smoke-check graph.
  --no-hf-cache            Do not mount the local Hugging Face cache into Docker services.
  --skip-build             Do not pass --build to docker compose up.
  -h, --help               Show this help.

Examples:
  scripts/gemma-2-2b-local.sh all
  scripts/gemma-2-2b-local.sh all --with-inference
  scripts/gemma-2-2b-local.sh all --webapp docker
  scripts/gemma-2-2b-local.sh up --graph mps
  scripts/gemma-2-2b-local.sh import --layers 20 --explanations only
  scripts/gemma-2-2b-local.sh import --layers 0-25 --import-jobs 2
  scripts/gemma-2-2b-local.sh verify

Environment:
  Docker with the Compose plugin is required for Postgres/db-init and Docker-managed services.
  Node.js and npm are required when using the default --webapp local mode.
  HF_TOKEN must be present in .env or apps/graph/.env before starting graph.
  HF_TOKEN must be present in .env before using --with-inference.
  ANTHROPIC_API_KEY must be present in apps/graph/.env before starting graph.
  GRAPH_PYTHON can override the MPS graph Python executable. Default: apps/graph/.venv/bin/python.
  GRAPH_SECRET and INFERENCE_SECRET default to localhost-secret.
EOF
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.."
}

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

print_log_tail() {
  local log_file="$1"
  local lines="${2:-60}"
  if [[ -f "${log_file}" ]]; then
    echo "---- last ${lines} lines of ${log_file} ----" >&2
    tail -n "${lines}" "${log_file}" >&2 || true
    echo "---- end ${log_file} ----" >&2
  fi
}

extract_transcoder_fetch_progress() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 0

  local line
  line="$(tr '\r' '\n' <"${log_file}" | grep -E 'Fetching [0-9]+ files:' | tail -n 1 || true)"
  [[ -n "${line}" ]] || return 0

  local total
  total="$(printf '%s' "${line}" | sed -n 's/.*Fetching \([0-9][0-9]*\) files:.*/\1/p')"
  local pair
  pair="$(printf '%s' "${line}" | grep -Eo '[0-9]+/[0-9]+' | tail -n 1 || true)"
  local percent
  percent="$(printf '%s' "${line}" | sed -n 's/.*Fetching [0-9][0-9]* files:[[:space:]]*\([0-9][0-9]*\)%.*/\1/p')"

  if [[ -n "${pair}" ]]; then
    local completed="${pair%%/*}"
    local detected_total="${pair##*/}"
    [[ -n "${total}" ]] || total="${detected_total}"
    [[ -n "${percent}" ]] || percent="$(awk -v done="${completed}" -v all="${total}" 'BEGIN { if (all > 0) printf "%.0f", done * 100 / all; }')"
    printf '%s/%s (%s%%)' "${completed}" "${total}" "${percent}"
  elif [[ -n "${percent}" && -n "${total}" ]]; then
    local completed
    completed="$(awk -v pct="${percent}" -v all="${total}" 'BEGIN { printf "%.0f", pct * all / 100; }')"
    printf '%s/%s (%s%%)' "${completed}" "${total}" "${percent}"
  fi
}

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  if (( hours > 0 )); then
    printf '%dh %dm %ds' "${hours}" "${minutes}" "${seconds}"
  elif (( minutes > 0 )); then
    printf '%dm %ds' "${minutes}" "${seconds}"
  else
    printf '%ds' "${seconds}"
  fi
}

report_all_duration() {
  local status="$1"
  local ended_at
  ended_at="$(date +%s)"
  local elapsed=$((ended_at - ALL_STARTED_AT))
  if [[ "${status}" == "0" ]]; then
    info "all completed in $(format_duration "${elapsed}")"
  else
    echo "error: all failed after $(format_duration "${elapsed}")" >&2
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_docker() {
  require_cmd docker
  docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is required; install a Docker version with 'docker compose'"
}

ensure_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    touch "${file}"
  fi
}

env_has_key() {
  local file="$1"
  local key="$2"
  [[ -f "${file}" ]] && grep -Eq "^[[:space:]]*${key}=" "${file}"
}

env_get() {
  local file="$1"
  local key="$2"
  if [[ ! -f "${file}" ]]; then
    return 1
  fi
  grep -E "^[[:space:]]*${key}=" "${file}" | tail -n 1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'
}

ensure_env_default() {
  local file="$1"
  local key="$2"
  local value="$3"
  ensure_file "${file}"
  if ! env_has_key "${file}" "${key}"; then
    printf '\n%s=%s\n' "${key}" "${value}" >>"${file}"
    info "added ${key} to ${file}"
  fi
}

write_webapp_runtime_env() {
  mkdir -p "${LOG_DIR}"
  [[ -f "${WEBAPP_ENV}" ]] || die "missing ${WEBAPP_ENV}; create it before using --webapp local"
  local runtime_env_dir
  runtime_env_dir="$(cd "$(dirname "${WEBAPP_RUNTIME_ENV}")" && pwd)"
  WEBAPP_RUNTIME_ENV_ABS="${runtime_env_dir}/$(basename "${WEBAPP_RUNTIME_ENV}")"

  grep -Ev '^(USE_LOCALHOST_GRAPH|GRAPH_SERVER_SECRET|USE_LOCALHOST_INFERENCE|INFERENCE_SERVER_SECRET)=' "${WEBAPP_ENV}" >"${WEBAPP_RUNTIME_ENV_ABS}"
  {
    printf '\nUSE_LOCALHOST_GRAPH=true\n'
    printf 'GRAPH_SERVER_SECRET=%s\n' "${GRAPH_SECRET}"
    if [[ "${START_INFERENCE}" == "1" ]]; then
      printf 'USE_LOCALHOST_INFERENCE=true\n'
    else
      printf 'USE_LOCALHOST_INFERENCE=false\n'
    fi
    printf 'INFERENCE_SERVER_SECRET=%s\n' "${INFERENCE_SECRET}"
  } >>"${WEBAPP_RUNTIME_ENV_ABS}"
}

require_hf_token() {
  local token=""
  token="$(env_get "${ROOT_ENV}" HF_TOKEN || true)"
  if [[ -z "${token}" ]]; then
    token="$(env_get "${GRAPH_ENV}" HF_TOKEN || true)"
  fi
  if [[ -z "${token}" || "${token}" == "your_hf_token" || "${token}" == "<your-hugging-face-token>" ]]; then
    die "set HF_TOKEN in ${ROOT_ENV} or ${GRAPH_ENV}; it needs access to ${TL_MODEL_ID}"
  fi
}

require_root_hf_token() {
  local token=""
  token="$(env_get "${ROOT_ENV}" HF_TOKEN || true)"
  if [[ -z "${token}" || "${token}" == "your_hf_token" || "${token}" == "<your-hugging-face-token>" ]]; then
    die "set HF_TOKEN in ${ROOT_ENV} before using --with-inference"
  fi
}

require_graph_anthropic_key() {
  local key=""
  key="$(env_get "${GRAPH_ENV}" ANTHROPIC_API_KEY || true)"
  if [[ -z "${key}" || "${key}" == "your_anthropic_api_key" || "${key}" == "<your-anthropic-api-key>" ]]; then
    die "set ANTHROPIC_API_KEY in ${GRAPH_ENV} before starting the graph server"
  fi
}

prepare_env() {
  ensure_env_default "${ROOT_ENV}" GRAPH_SERVER_SECRET "${GRAPH_SECRET}"
  ensure_env_default "${ROOT_ENV}" INFERENCE_SERVER_SECRET "${INFERENCE_SECRET}"
  ensure_env_default "${LOCALHOST_ENV}" GRAPH_SERVER_SECRET "${GRAPH_SECRET}"
  ensure_env_default "${LOCALHOST_ENV}" INFERENCE_SERVER_SECRET "${INFERENCE_SECRET}"
  ensure_env_default "${LOCALHOST_ENV}" USE_LOCALHOST_GRAPH "true"
  if [[ "${START_INFERENCE}" == "1" ]]; then
    ensure_env_default "${LOCALHOST_ENV}" USE_LOCALHOST_INFERENCE "true"
  fi

  ensure_env_default "${GRAPH_ENV}" GRAPH_SERVER_SECRET "${GRAPH_SECRET}"
  ensure_env_default "${GRAPH_ENV}" SECRET "${GRAPH_SECRET}"
  ensure_env_default "${GRAPH_ENV}" MODEL_ID "${TL_MODEL_ID}"
  ensure_env_default "${GRAPH_ENV}" TRANSCODER_SET "${TRANSCODER_SET}"
  ensure_env_default "${GRAPH_ENV}" ANTHROPIC_API_KEY "<your-anthropic-api-key>"
}

compose_base() {
  docker compose \
    -f docker/compose.yaml \
    --env-file "${LOCALHOST_ENV}" \
    --env-file "${ROOT_ENV}" \
    "$@"
}

compose_db_up() {
  local build_arg=()
  if [[ "${SKIP_BUILD}" != "1" ]]; then
    build_arg=(--build)
  fi

  info "starting Postgres and db-init"
  compose_base up -d "${build_arg[@]}" postgres db-init
}

webapp_docker_up() {
  [[ "${WEBAPP_MODE}" == "docker" ]] || return 0

  local build_arg=()
  if [[ "${SKIP_BUILD}" != "1" ]]; then
    build_arg=(--build)
  fi

  info "starting webapp through Docker production image"
  compose_base up -d "${build_arg[@]}" webapp
}

webapp_local_up() {
  [[ "${WEBAPP_MODE}" == "local" ]] || return 0
  require_cmd npm

  mkdir -p "${LOG_DIR}"
  if curl --fail --silent --output /dev/null "${WEB_URL}"; then
    info "webapp already responds at ${WEB_URL}"
    return 0
  fi

  if [[ ! -d apps/webapp/node_modules ]]; then
    info "installing webapp npm dependencies"
    (cd apps/webapp && npm install)
  fi

  write_webapp_runtime_env

  if [[ -f "${WEBAPP_PID_FILE}" ]]; then
    local old_pid
    old_pid="$(cat "${WEBAPP_PID_FILE}")"
    if kill -0 "${old_pid}" >/dev/null 2>&1; then
      info "local webapp already running as pid ${old_pid}"
      return 0
    fi
  fi

  info "starting local webapp; log: ${WEBAPP_LOG}"
  (
    cd apps/webapp
    npx env-cmd -f "${WEBAPP_RUNTIME_ENV_ABS}" --use-shell "prisma generate && next dev"
  ) >"${WEBAPP_LOG}" 2>&1 &
  echo "$!" >"${WEBAPP_PID_FILE}"
}

compose_inference_up() {
  [[ "${START_INFERENCE}" == "1" ]] || return 0
  require_root_hf_token

  local files=(-f docker/compose.yaml -f docker/compose.inference.dev.yaml)
  if [[ "${USE_CUDA}" == "1" ]]; then
    files+=(-f docker/compose.inference.gpu.yaml)
  fi
  if [[ "${USE_LOCAL_HF_CACHE}" == "1" ]]; then
    files+=(-f docker/compose.hf-cache.yaml)
  fi

  local build_arg=()
  if [[ "${SKIP_BUILD}" != "1" ]]; then
    build_arg=(--build)
  fi

  info "starting inference server for ${MODEL_SOURCESET}"
  ENV_FILE="../${INFERENCE_ENV}" \
    docker compose \
      "${files[@]}" \
      --env-file "${INFERENCE_ENV}" \
      --env-file "${LOCALHOST_ENV}" \
      --env-file "${ROOT_ENV}" \
      up -d "${build_arg[@]}" inference
}

compose_graph_up() {
  [[ "${START_GRAPH}" == "1" ]] || return 0
  [[ "${GRAPH_MODE}" == "docker" ]] || return 0
  require_hf_token
  require_graph_anthropic_key

  local files=(-f docker/compose.yaml -f docker/compose.graph.dev.yaml)
  if [[ "${USE_CUDA}" == "1" ]]; then
    files+=(-f docker/compose.graph.gpu.yaml)
  fi
  if [[ "${USE_LOCAL_HF_CACHE}" == "1" ]]; then
    files+=(-f docker/compose.hf-cache.yaml)
  fi

  local build_arg=()
  if [[ "${SKIP_BUILD}" != "1" ]]; then
    build_arg=(--build)
  fi

  info "starting graph server for ${TL_MODEL_ID} / ${TRANSCODER_SET}"
  ENV_FILE="${GRAPH_ENV}" MODEL_ID="${TL_MODEL_ID}" \
    docker compose \
      --project-directory . \
      "${files[@]}" \
      --env-file "${GRAPH_ENV}" \
      up -d "${build_arg[@]}" graph
}

graph_mps_up() {
  [[ "${START_GRAPH}" == "1" ]] || return 0
  [[ "${GRAPH_MODE}" == "mps" ]] || return 0
  [[ -x "${GRAPH_PYTHON}" ]] || die "missing executable graph Python at ${GRAPH_PYTHON}; run 'make graph-localhost-install' or set GRAPH_PYTHON"
  local graph_python_abs
  graph_python_abs="$(cd "$(dirname "${GRAPH_PYTHON}")" && pwd)/$(basename "${GRAPH_PYTHON}")"
  require_hf_token
  require_graph_anthropic_key

  mkdir -p "${LOG_DIR}"
  if [[ -f "${GRAPH_MPS_PID_FILE}" ]]; then
    local old_pid
    old_pid="$(cat "${GRAPH_MPS_PID_FILE}")"
    if kill -0 "${old_pid}" >/dev/null 2>&1; then
      info "MPS graph server already running as pid ${old_pid}"
      return 0
    fi
  fi

  local hf_token
  hf_token="$(env_get "${GRAPH_ENV}" HF_TOKEN || true)"
  if [[ -z "${hf_token}" ]]; then
    hf_token="$(env_get "${ROOT_ENV}" HF_TOKEN || true)"
  fi
  local anthropic_key
  anthropic_key="$(env_get "${GRAPH_ENV}" ANTHROPIC_API_KEY)"

  info "starting graph server with MPS; log: ${GRAPH_MPS_LOG}"
  (
    cd apps/graph
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
      SECRET="${GRAPH_SECRET}" \
      GRAPH_SERVER_SECRET="${GRAPH_SECRET}" \
      HF_TOKEN="${hf_token}" \
      ANTHROPIC_API_KEY="${anthropic_key}" \
      TRANSCODER_SET="${TRANSCODER_SET}" \
      "${graph_python_abs}" start.py \
        --model_id "${TL_MODEL_ID}" \
        --transcoder_set "${TRANSCODER_SET}" \
        --device mps \
        --port 5004
  ) >"${GRAPH_MPS_LOG}" 2>&1 &
  echo "$!" >"${GRAPH_MPS_PID_FILE}"
}

wait_url() {
  local url="$1"
  local label="$2"
  local timeout="${3:-180}"
  local pid_file="${4:-}"
  local log_file="${5:-}"
  local start
  start="$(date +%s)"
  local last_progress="${start}"
  local last_transcoder_progress=""

  info "waiting for ${label} at ${url}"
  until curl --fail --silent --output /dev/null "${url}"; do
    local now
    now="$(date +%s)"
    if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        print_log_tail "${log_file}" 80
        die "${label} process exited before ${url} became ready"
      fi
    fi
    if [[ -n "${log_file}" && "${label}" == "graph server" ]]; then
      local transcoder_progress
      transcoder_progress="$(extract_transcoder_fetch_progress "${log_file}")"
      if [[ -n "${transcoder_progress}" && "${transcoder_progress}" != "${last_transcoder_progress}" ]]; then
        info "transcoder layers downloaded: ${transcoder_progress}"
        last_transcoder_progress="${transcoder_progress}"
      fi
    fi
    if (( now - last_progress >= 30 )); then
      local elapsed=$((now - start))
      info "still waiting for ${label} after $(format_duration "${elapsed}")"
      if [[ -n "${log_file}" ]]; then
        print_log_tail "${log_file}" 20
      fi
      last_progress="${now}"
    fi
    if (( now - start > timeout )); then
      if [[ -n "${log_file}" ]]; then
        print_log_tail "${log_file}" 80
      fi
      die "${label} did not become ready within ${timeout}s: ${url}"
    fi
    sleep 3
  done
}

start_services() {
  prepare_env
  require_docker
  require_cmd curl

  compose_db_up
  webapp_docker_up
  webapp_local_up
  compose_inference_up
  compose_graph_up
  graph_mps_up
}

smoke() {
  require_cmd curl
  if [[ "${WEBAPP_MODE}" != "skip" ]]; then
    wait_url "${WEB_URL}" "webapp" 240
  fi
  if [[ "${START_INFERENCE}" == "1" ]]; then
    wait_url "${INFERENCE_URL}/docs" "inference server" 600
  fi
  if [[ "${START_GRAPH}" == "1" ]]; then
    if [[ "${GRAPH_MODE}" == "mps" ]]; then
      wait_url "${GRAPH_URL}/docs" "graph server" 900 "${GRAPH_MPS_PID_FILE}" "${GRAPH_MPS_LOG}"
    else
      wait_url "${GRAPH_URL}/docs" "graph server" 900
    fi
  fi
}

expand_layers() {
  local spec="$1"
  local part
  local -a result=()
  IFS=',' read -ra parts <<<"${spec}"
  for part in "${parts[@]}"; do
    if [[ "${part}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      if (( start > end )); then
        die "invalid descending layer range: ${part}"
      fi
      local layer
      for ((layer = start; layer <= end; layer++)); do
        result+=("${layer}")
      done
    elif [[ "${part}" =~ ^[0-9]+$ ]]; then
      result+=("${part}")
    else
      die "invalid layer spec: ${part}"
    fi
  done
  printf '%s\n' "${result[@]}"
}

import_one_source() {
  local source_id="$1"
  local url="${WEB_URL}/api/admin/import?modelId=${MODEL_ID}&sourceId=${source_id}&explanations=${EXPLANATIONS}"
  local output_file
  output_file="$(mktemp)"
  local started_at
  started_at="$(date +%s)"

  info "importing ${MODEL_ID}/${source_id} with explanations=${EXPLANATIONS}"
  if ! curl --fail --silent --show-error --no-buffer "${url}" | tee "${output_file}" | while IFS= read -r line; do
    if [[ "${line}" != data:\ * ]]; then
      continue
    fi
    local payload="${line#data: }"
    local progress_text
    progress_text="$(printf '%s' "${payload}" | sed -n 's/.*"progressText":"\([^"]*\)".*/\1/p')"
    local progress
    progress="$(printf '%s' "${payload}" | sed -n 's/.*"progress":\([0-9.]*\).*/\1/p')"
    if [[ -n "${progress_text}" ]]; then
      if [[ -n "${progress}" ]]; then
        local percent
        percent="$(awk -v value="${progress}" 'BEGIN { printf "%.1f%%", value * 100 }')"
        printf '==> [%s] %s (%s)\n' "${source_id}" "${progress_text}" "${percent}"
      else
        printf '==> [%s] %s\n' "${source_id}" "${progress_text}"
      fi
    fi
  done; then
    cat "${output_file}" >&2 || true
    rm -f "${output_file}"
    die "import failed for ${source_id}"
  fi

  if grep -q '^event: error' "${output_file}"; then
    cat "${output_file}" >&2
    rm -f "${output_file}"
    die "import endpoint returned an error for ${source_id}"
  fi

  rm -f "${output_file}"
  local elapsed
  elapsed=$(( $(date +%s) - started_at ))
  info "imported ${MODEL_ID}/${source_id} in $(format_duration "${elapsed}")"
}

import_data() {
  require_cmd curl
  wait_url "${WEB_URL}" "webapp" 240
  [[ "${IMPORT_JOBS}" =~ ^[0-9]+$ && "${IMPORT_JOBS}" -ge 1 ]] || die "IMPORT_JOBS must be a positive integer"

  local -a source_ids=()
  local layer
  while IFS= read -r layer; do
    source_ids+=("${layer}-${SOURCE_SET}")
  done < <(expand_layers "${LAYERS_SPEC}")

  info "importing ${#source_ids[@]} source(s) with ${IMPORT_JOBS} concurrent job(s)"
  if (( IMPORT_JOBS == 1 || ${#source_ids[@]} <= 1 )); then
    local source_id
    for source_id in "${source_ids[@]}"; do
      import_one_source "${source_id}"
    done
    return
  fi

  local failed=0
  local source_id
  local -a import_pids=()
  local -a import_pid_sources=()

  cleanup_import_jobs() {
    local pid
    for pid in "${import_pids[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
  }

  wait_for_oldest_import() {
    local pid="${import_pids[0]}"
    local waited_source="${import_pid_sources[0]}"
    if ! wait "${pid}"; then
      echo "error: import failed for ${waited_source}" >&2
      failed=1
      cleanup_import_jobs
    fi
    import_pids=("${import_pids[@]:1}")
    import_pid_sources=("${import_pid_sources[@]:1}")
  }

  trap cleanup_import_jobs INT TERM
  for source_id in "${source_ids[@]}"; do
    if (( failed != 0 )); then
      break
    fi
    import_one_source "${source_id}" &
    import_pids+=("$!")
    import_pid_sources+=("${source_id}")
    if (( ${#import_pids[@]} >= IMPORT_JOBS )); then
      wait_for_oldest_import
    fi
  done

  while (( ${#import_pids[@]} > 0 )); do
    wait_for_oldest_import
  done
  trap - INT TERM

  if (( failed != 0 )); then
    die "one or more source imports failed"
  fi
}

verify_data() {
  require_docker
  info "printing Gemma-2-2B import counts"
  compose_base exec -T postgres psql -U postgres -d postgres <<SQL
SELECT count(*) AS sources
FROM "Source"
WHERE "modelId" = '${MODEL_ID}'
  AND "setName" = '${SOURCE_SET}';

SELECT count(*) AS features
FROM "Neuron"
WHERE "modelId" = '${MODEL_ID}'
  AND "sourceSetName" = '${SOURCE_SET}';

SELECT count(*) AS activations
FROM "Activation"
WHERE "modelId" = '${MODEL_ID}'
  AND "layer" LIKE '%${SOURCE_SET}';

SELECT count(*) AS explanations
FROM "Explanation" e
JOIN "Neuron" n
  ON n."modelId" = e."modelId"
 AND n."layer" = e."layer"
 AND n."index" = e."index"
WHERE n."modelId" = '${MODEL_ID}'
  AND n."sourceSetName" = '${SOURCE_SET}';
SQL
}

stop_services() {
  require_docker
  info "stopping Docker services without deleting volumes"
  docker compose \
    -f docker/compose.yaml \
    -f docker/compose.inference.dev.yaml \
    -f docker/compose.graph.dev.yaml \
    -f docker/compose.hf-cache.yaml \
    --env-file "${LOCALHOST_ENV}" \
    --env-file "${ROOT_ENV}" \
    down

  if [[ -f "${GRAPH_MPS_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${GRAPH_MPS_PID_FILE}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      info "stopping MPS graph server pid ${pid}"
      kill "${pid}"
    fi
    rm -f "${GRAPH_MPS_PID_FILE}"
  fi

  if [[ -f "${WEBAPP_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${WEBAPP_PID_FILE}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      info "stopping local webapp pid ${pid}"
      kill "${pid}"
    fi
    rm -f "${WEBAPP_PID_FILE}"
  fi
}

run_all() {
  ALL_STARTED_AT="$(date +%s)"
  info "all started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  start_services
  smoke
  import_data
  verify_data
  report_all_duration 0
  ALL_STARTED_AT=0
}

parse_args() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  COMMAND="$1"
  shift

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --layers)
        LAYERS_SPEC="${2:-}"
        [[ -n "${LAYERS_SPEC}" ]] || die "--layers requires a value"
        shift 2
        ;;
      --import-jobs)
        IMPORT_JOBS="${2:-}"
        [[ "${IMPORT_JOBS}" =~ ^[0-9]+$ && "${IMPORT_JOBS}" -ge 1 ]] || die "--import-jobs requires a positive integer"
        shift 2
        ;;
      --explanations)
        EXPLANATIONS="${2:-}"
        [[ "${EXPLANATIONS}" == "true" || "${EXPLANATIONS}" == "false" || "${EXPLANATIONS}" == "only" ]] || die "--explanations must be true, false, or only"
        shift 2
        ;;
      --cuda)
        USE_CUDA=1
        shift
        ;;
      --with-inference)
        START_INFERENCE=1
        shift
        ;;
      --webapp)
        WEBAPP_MODE="${2:-}"
        [[ "${WEBAPP_MODE}" == "local" || "${WEBAPP_MODE}" == "docker" || "${WEBAPP_MODE}" == "skip" ]] || die "--webapp must be local, docker, or skip"
        shift 2
        ;;
      --graph)
        GRAPH_MODE="${2:-}"
        [[ "${GRAPH_MODE}" == "docker" || "${GRAPH_MODE}" == "mps" || "${GRAPH_MODE}" == "skip" ]] || die "--graph must be docker, mps, or skip"
        if [[ "${GRAPH_MODE}" == "skip" ]]; then
          START_GRAPH=0
        fi
        shift 2
        ;;
      --no-inference)
        START_INFERENCE=0
        shift
        ;;
      --no-graph)
        START_GRAPH=0
        GRAPH_MODE="skip"
        shift
        ;;
      --no-hf-cache)
        USE_LOCAL_HF_CACHE=0
        shift
        ;;
      --skip-build)
        SKIP_BUILD=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
  done
}

main() {
  repo_root
  parse_args "$@"

  case "${COMMAND}" in
    up)
      start_services
      smoke
      ;;
    import)
      import_data
      ;;
    verify)
      verify_data
      ;;
    smoke)
      smoke
      ;;
    all)
      run_all
      ;;
    down)
      stop_services
      ;;
    -h|--help)
      usage
      ;;
    *)
      usage
      die "unknown command: ${COMMAND}"
      ;;
  esac
}

on_exit() {
  local status="$?"
  if [[ "${ALL_STARTED_AT:-0}" != "0" ]]; then
    report_all_duration "${status}"
  fi
}

trap on_exit EXIT

main "$@"
