#!/bin/bash

#SBATCH --job-name=experiment_ordered_queries_pg19
#SBATCH -c 20
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=experiment_ordered_queries_pg19/slurm-%j.out
#SBATCH --error=experiment_ordered_queries_pg19/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Running on node: $SLURMD_NODENAME"
echo "Current directory before cd: $(pwd)"
echo "Start time: $(date)"

# Ir al directorio principal del experimento PG19
cd ~/RAG-timing-pg19

echo "Current directory after cd: $(pwd)"

# Activar el entorno virtual de PG19
source venv/bin/activate

echo "Python being used:"
which python
python --version

echo "Checking GPU..."
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY

echo "Checking required paths..."
ls -lh pg19-queries.json
ls -lh workspace/pg19
ls -lh experiment_ordered_queries_pg19

echo "Running combined retrieval-only and full RAG timing experiment for PG19..."
python experiment_ordered_queries_pg19/run_pg19_experiment4_ordered.py \
  --queries pg19_queries.json \
  --database workspace/pg19 \
  --collection-name pg19 \
  --output-root experiment_ordered_queries_pg19/results_pg19_combined \
  --k-values 5,20,50,100 \
  --retrieval-iterations 2 \
  --rag-iterations 2 \
  --seed 42 \
  --embedding-model all-MiniLM-L6-v2 \
  --llm-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --mode both

echo "Job finished"
echo "End time: $(date)"
