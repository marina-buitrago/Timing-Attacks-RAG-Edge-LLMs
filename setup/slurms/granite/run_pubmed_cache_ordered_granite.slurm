#!/bin/bash
#SBATCH --job-name=experiment_cache_pubmed_granite
#SBATCH -c 20
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=experiment-cache-granite-pubmed/slurm-granite-%j.out
#SBATCH --error=experiment-cache-granite-pubmed/slurm-granite-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

cd "$HOME/RAG-timing"
source venv/bin/activate

SCRIPT="experiment-cache-granite-pubmed/run_pubmed_cache_ordered_granite.py"
OUTPUT_ROOT="experiment-cache-granite-pubmed/results_pubmed_cache_granite"
REDIS_SERVER="/usr/bin/redis-server"
REDIS_MODULE_DIR="/usr/lib/redis/modules"
REDIS_LOG="$PWD/experiment-cache-granite-pubmed/redis-${SLURM_JOB_ID}.log"

mkdir -p experiment-cache-granite-pubmed

if [ ! -x "$REDIS_SERVER" ]; then
    echo "ERROR: Redis is not installed on this compute node." >&2
    exit 1
fi

for file in \
    "$SCRIPT" \
    "queries-3-pubmed.json" \
    "$REDIS_MODULE_DIR/rejson.so" \
    "$REDIS_MODULE_DIR/redisearch.so"
do
    if [ ! -r "$file" ]; then
        echo "ERROR: Required file missing or unreadable: $file" >&2
        exit 1
    fi
done

ls -ld workspace/pubmed_central

if [ -d "$OUTPUT_ROOT" ] && [ -n "$(ls -A "$OUTPUT_ROOT")" ]; then
    echo "ERROR: The results folder is not empty: $OUTPUT_ROOT" >&2
    echo "Back it up before submitting this job again." >&2
    exit 1
fi

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Python: $(command -v python)"
python --version
"$REDIS_SERVER" --version

python - <<'PY'
import torch
import redis
from langchain_redis import RedisSemanticCache

print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Redis Python packages: OK")
PY

python -m pip freeze > "experiment-cache-granite-pubmed/packages-${SLURM_JOB_ID}.txt"

REDIS_DIR=$(mktemp -d "/tmp/pubmed-redis.${SLURM_JOB_ID}.XXXXXX")
export REDIS_URL="unix://${REDIS_DIR}/redis.sock?db=0"
REDIS_PID=""

cleanup() {
    local status=$?
    trap - EXIT

    if [ -n "${REDIS_PID:-}" ]; then
        kill "$REDIS_PID" 2>/dev/null || true
        wait "$REDIS_PID" 2>/dev/null || true
    fi

    rm -rf -- "$REDIS_DIR"
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting Redis directly; log: $REDIS_LOG"

"$REDIS_SERVER" \
    --loadmodule "$REDIS_MODULE_DIR/rejson.so" \
    --loadmodule "$REDIS_MODULE_DIR/redisearch.so" \
    --port 0 \
    --protected-mode yes \
    --unixsocket "$REDIS_DIR/redis.sock" \
    --unixsocketperm 700 \
    --dir "$REDIS_DIR" \
    --save "" \
    --appendonly no \
    --daemonize no \
    --logfile "" \
    > "$REDIS_LOG" 2>&1 &

REDIS_PID=$!
export REDIS_PID

echo "Checking Redis and Search support..."

if ! python - <<'PY'
import os
import time
import redis

client = redis.Redis.from_url(
    os.environ["REDIS_URL"],
    socket_connect_timeout=2,
    socket_timeout=2,
    decode_responses=True,
)

deadline = time.monotonic() + 60

try:
    while True:
        try:
            client.ping()
            client.execute_command("FT._LIST")

            print("Redis is ready: Search support detected.")
            print("Redis server version:", client.info("server")["redis_version"])
            print("Loaded modules:", client.module_list())
            break

        except redis.exceptions.ResponseError as exc:
            raise SystemExit(
                f"Redis Search check failed: {exc}"
            ) from exc

        except (
            redis.exceptions.ConnectionError,
            redis.exceptions.TimeoutError,
        ):
            try:
                os.kill(int(os.environ["REDIS_PID"]), 0)
            except ProcessLookupError:
                raise SystemExit(
                    "Redis exited before becoming ready."
                )

            if time.monotonic() >= deadline:
                raise SystemExit(
                    "Redis did not become ready."
                )

            time.sleep(1)

finally:
    client.close()
PY
then
    echo "ERROR: Redis startup failed. Redis log follows:" >&2
    tail -n 80 "$REDIS_LOG" >&2
    exit 1
fi

echo "Running retrieval-only and full RAG with semantic LLM caching..."

python "$SCRIPT" \
    --project-root "$PWD" \
    --queries queries-3-pubmed.json \
    --database workspace/pubmed_central \
    --collection-name pubmed_central \
    --output-root "$OUTPUT_ROOT" \
    --k-values 5,20,50,100 \
    --retrieval-iterations 2 \
    --rag-iterations 2 \
    --seed 42 \
    --embedding-model all-MiniLM-L6-v2 \
    --llm-model ibm-granite/granite-3.0-1b-a400m-instruct \
    --mode both \
    --semantic-cache \
    --cache-name pubmed_llm_semantic \
    --cache-threshold 0.1 \
    --cache-reset per-k \
    --cache-ttl 0

echo "Results: $OUTPUT_ROOT"
echo "Job finished: $(date)"
