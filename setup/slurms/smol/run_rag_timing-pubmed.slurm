#!/bin/bash

#SBATCH --job-name=experiment_ordered_queries_pubmed
#SBATCH -c 20
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=experiment_ordered_queries_pubmed/slurm-%j.out
#SBATCH --error=experiment_ordered_queries_pubmed/slurm-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Running on node: $SLURMD_NODENAME"
echo "Current directory before cd: $(pwd)"
echo "Start time: $(date)"

# Go to the main experiment folder.
cd ~/RAG-timing

echo "Current directory after cd: $(pwd)"

# Activate the existing virtual environment.
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
ls -lh queries-3-pubmed.json
ls -lh workspace/pubmed_central
ls -lh experiment_ordered_queries_pubmed

echo "Running combined retrieval-only and full RAG timing experiment..."
python experiment_ordered_queries_pubmed/run_pubmed_combined_latency_experiment_ordered.py \
  --queries queries-3-pubmed.json \
  --database workspace/pubmed_central \
  --collection-name pubmed_central \
  --output-root experiment_ordered_queries_pubmed/results_pubmed_combined_ordered \
  --k-values 5,20,50,100 \
  --retrieval-iterations 2 \
  --rag-iterations 2 \
  --seed 42 \
  --embedding-model all-MiniLM-L6-v2 \
  --llm-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --mode both

echo "Job finished"
echo "End time: $(date)"

