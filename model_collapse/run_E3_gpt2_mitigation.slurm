#!/bin/bash
#SBATCH --job-name=E3_level1_mitigation
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --export=NONE
#SBATCH --output=E3_level1_mitigation.o%j
#SBATCH --error=E3_level1_mitigation.e%j

set -euo pipefail
unset SLURM_EXPORT_ENV

echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-N/A}"

# -----------------------------------------------------------------------------
# Robust WORKDIR:
# - If you submit with:  sbatch run_E3_gpt2_mitigation.slurm   from inside model_collapse/
#   then SLURM_SUBMIT_DIR is perfect.
# - Fallback: directory where this .slurm file lives.
# -----------------------------------------------------------------------------
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CONTAINER="$HOME/containers/python311.sif"

cd "$WORKDIR"
echo "[INFO] Working dir: $(pwd)"

# Fixed run folder per job
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
echo "[INFO] RUN_DIR: $RUN_DIR"

# -----------------------------------------------------------------------------
# Caches / offline / wandb
# -----------------------------------------------------------------------------
export HF_HOME="$WORKDIR/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Project-local python userbase (persists across jobs on $HOME)
export PYTHONUSERBASE="$WORKDIR/.pyuserbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
for f in data/wikitext2/train.json data/wikitext2/validation.json data/wikitext2/test.json; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing dataset file: $f"
    exit 1
  fi
done
echo "[INFO] Human dataset JSONs found."

# -----------------------------------------------------------------------------
# Write dataset config expected by main.py (Hydra)
# -----------------------------------------------------------------------------
mkdir -p config/dataset
cat > config/dataset/wikitext2_local.yaml << 'EOF'
path: data/wikitext2
EOF
echo "[INFO] Wrote config/dataset/wikitext2_local.yaml"

# -----------------------------------------------------------------------------
# Install deps ONCE (huge speed-up vs installing every job)
# -----------------------------------------------------------------------------
DEPS_STAMP="$WORKDIR/.deps_installed_ok"
if [ ! -f "$DEPS_STAMP" ]; then
  echo "[INFO] Installing Python packages (first time for this WORKDIR) ..."
  singularity exec --nv "$CONTAINER" bash -lc "
    set -e
    cd '$WORKDIR'
    export PYTHONUSERBASE='$PYTHONUSERBASE'
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"

    python -m pip install --upgrade --user pip setuptools wheel

    # Extra deps that are often missing
    python -m pip install --user \
      'torch>=2.0' \
      'transformers>=4.40' \
      'datasets>=2.14' \
      'accelerate>=0.20' \
      sentencepiece protobuf

    # Project requirements
    python -m pip install --user -r requirements.txt

    python - << 'EOF'
import transformers, torch, datasets, accelerate
print('transformers', transformers.__version__)
print('torch', torch.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
EOF
  "
  touch "$DEPS_STAMP"
  echo "[INFO] Dependency install completed; stamp written: $DEPS_STAMP"
else
  echo "[INFO] Dependencies already installed (found stamp: $DEPS_STAMP). Skipping pip install."
fi

# -----------------------------------------------------------------------------
# Precreate RUN_DIR/<i>/data.json expected by the repo
# -----------------------------------------------------------------------------
NUM_ITERS=10
for i in $(seq 0 $NUM_ITERS); do
  mkdir -p "$RUN_DIR/$i"
  if [ ! -f "$RUN_DIR/$i/data.json" ]; then
    cp -f "data/wikitext2/train.json" "$RUN_DIR/$i/data.json"
  fi
done
echo "[INFO] Ensured $((NUM_ITERS+1)) generation folders with data.json"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
echo "[INFO] Starting main.py ..."
singularity exec --nv "$CONTAINER" bash -lc "
  set -e
  cd '$WORKDIR'
  export HF_HOME='$HF_HOME'
  export TRANSFORMERS_OFFLINE=1
  export WANDB_DISABLED=true
  export PYTHONUSERBASE='$PYTHONUSERBASE'
  export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"

  python main.py \
    hydra.run.dir='$RUN_DIR' \
    dataset=wikitext2_local \
    model=gpt2 \
    decoding=top_k \
    detector=modernbert_mage \
    data_selection=importance_sampling \
    num_iterations=10 \
    wandb_disabled=true
"

echo "[INFO] Done at $(date)"
