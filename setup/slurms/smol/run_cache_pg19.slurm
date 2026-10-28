#!/bin/bash
#SBATCH --job-name=experiment_cache_pg19
#SBATCH -c 20
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=experiment_cache_pg19/slurm-%j.out
#SBATCH --error=experiment_cache_pg19/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# Submit this file from ~/RAG-timing-pg19.
cd "$HOME/RAG-timing-pg19"

# Activate the existing virtual environment.
source venv/bin/activate

# If your cluster requires an Apptainer module, load it here using the
# module command provided by cluster support.
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: Apptainer is not available on this compute node." >&2
    echo "Ask cluster support how to enable Apptainer for this job." >&2
    exit 1
fi

# Use your existing Python filename and results folder.
SCRIPT="experiment_cache_pg19/run_pg19_experiment4_ordered.py"
OUTPUT_ROOT="experiment_cache_pg19/results_pg19_combined_ordered"

# Redis image and job-specific Redis log.
REDIS_IMAGE="$PWD/containers/redis-8.4.6.sif"
REDIS_LOG="$PWD/experiment_cache_pg19/redis-${SLURM_JOB_ID}.log"

echo "Checking required paths..."
ls -ld pg19-queries.json workspace/pg19 "$SCRIPT" "$REDIS_IMAGE"

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Current directory: $PWD"
echo "Python being used: $(command -v python)"
python --version

echo "Checking GPU and cache packages..."
python - <<'PY'
import torch
import redis
from langchain_redis import RedisSemanticCache

print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Redis Python packages imported successfully.")
PY

# Redis runs only for this job, on this compute node.
# A private local socket avoids opening a network port.
REDIS_DIR=$(mktemp -d "/tmp/pg19-redis.${SLURM_JOB_ID}.XXXXXX")
export REDIS_URL="unix://${REDIS_DIR}/redis.sock?db=0"
REDIS_PID=""

cleanup() {
    if [ -n "${REDIS_PID:-}" ]; then
        kill "$REDIS_PID" 2>/dev/null || true
        wait "$REDIS_PID" 2>/dev/null || true
    fi
    rm -rf "$REDIS_DIR"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting Redis; log: $REDIS_LOG"

# Use the image's entrypoint to start Redis with its modules.
apptainer exec --cleanenv \
    --bind "$REDIS_DIR:$REDIS_DIR" \
    "$REDIS_IMAGE" \
    /usr/local/bin/docker-entrypoint.sh redis-server \
    --port 0 \
    --unixsocket "$REDIS_DIR/redis.sock" \
    --unixsocketperm 700 \
    --dir "$REDIS_DIR" \
    --save "" \
    --appendonly no \
    --daemonize no \
    > "$REDIS_LOG" 2>&1 &

REDIS_PID=$!

echo "Checking that Redis is ready..."
python - <<'PY'
import os
import time
import redis

client = redis.Redis.from_url(
    os.environ["REDIS_URL"],
    socket_connect_timeout=2,
    socket_timeout=2,
)

deadline = time.monotonic() + 60

try:
    while True:
        try:
            client.ping()
            client.execute_command("FT._LIST")
            print("Redis is ready: Search support detected.")
            break

        except redis.exceptions.ResponseError as exc:
            raise SystemExit(
                f"Redis Search check failed: {exc}"
            ) from exc

        except (
            redis.exceptions.ConnectionError,
            redis.exceptions.TimeoutError,
        ):
            if time.monotonic() >= deadline:
                raise SystemExit(
                    "Redis did not start. Check "
                    "experiment_cache_pg19/redis-"
                    + os.environ["SLURM_JOB_ID"]
                    + ".log"
                )
            time.sleep(1)
finally:
    client.close()
PY

echo "Running retrieval-only and full RAG with semantic LLM caching..."

python "$SCRIPT" \
    --project-root "$PWD" \
    --queries pg19-queries.json \
    --database workspace/pg19 \
    --collection-name pg19 \
    --output-root "$OUTPUT_ROOT" \
    --k-values 5,20,50,100 \
    --retrieval-iterations 2 \
    --rag-iterations 2 \
    --seed 42 \
    --embedding-model all-MiniLM-L6-v2 \
    --llm-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --mode both \
    --semantic-cache \
    --cache-threshold 0.1 \
    --cache-reset per-k \
    --cache-ttl 0

echo "Results: $OUTPUT_ROOT"
echo "Job finished: $(date)"
