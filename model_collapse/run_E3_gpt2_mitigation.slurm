#!/bin/bash
#SBATCH --job-name=E3_level1_mitigation
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
#SBATCH --output=E3_level1_mitigation.o%j
#SBATCH --error=E3_level1_mitigation.e%j

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"

WORKDIR="/home/hpc/b116ba/b117ba41/Masterarbeit/Masterarbeit-firstSteps/TestPaper/KonkreteVersuche/E3_drayson/Machine-generated_text_detection_prevents_language_model_collapse/model_collapse"
CONTAINER="$HOME/containers/python311.sif"

# fester Run-Ordner pro Job
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"

cd "$WORKDIR"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] RUN_DIR: $RUN_DIR"

# --- Dataset sanity check ---
for f in data/wikitext2/train.json data/wikitext2/validation.json data/wikitext2/test.json; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing dataset file: $f"
    exit 1
  fi
done
echo "[INFO] Human dataset JSONs found."

# --- caches / offline / wandb ---
export HF_HOME="$WORKDIR/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# --- project-local python userbase ---
export PYTHONUSERBASE="$WORKDIR/.pyuserbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"

# --- Dataset config: main.py expects cfg.dataset.path ---
mkdir -p config/dataset
cat > config/dataset/wikitext2_local.yaml << 'EOF'
path: data/wikitext2
EOF
echo "[INFO] Wrote config/dataset/wikitext2_local.yaml"

# --- install deps (transformers missing in requirements) ---
echo "[INFO] Installing Python packages ..."
singularity exec --nv "$CONTAINER" bash -lc "
  set -e
  cd '$WORKDIR'
  export PYTHONUSERBASE='$PYTHONUSERBASE'
  export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"

  python -m pip install --upgrade --user pip setuptools wheel

  python -m pip install --user \
    'torch>=2.0' \
    'transformers>=4.40' \
    'datasets>=2.14' \
    'accelerate>=0.20' \
    sentencepiece protobuf

  python -m pip install --user -r requirements.txt

  python - << 'EOF'
import transformers, torch, datasets, accelerate
print('transformers', transformers.__version__)
print('torch', torch.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
EOF
"

# --- PRECREATE data.json structure expected by the repo ---
# The repo tries to read RUN_DIR/<i>/data.json. We create 0..10.
# For iteration 0 we use human train.json; for now we copy the same file as placeholder so the pipeline won't crash.
# This makes the run executable; later we can switch placeholder generations to real synthetic outputs once generation is confirmed.
NUM_ITERS=10
mkdir -p "$RUN_DIR"
for i in $(seq 0 $NUM_ITERS); do
  mkdir -p "$RUN_DIR/$i"
  # Create expected file name
  cp -f "data/wikitext2/train.json" "$RUN_DIR/$i/data.json"
done
echo "[INFO] Precreated $((NUM_ITERS+1)) generation folders with data.json"

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

