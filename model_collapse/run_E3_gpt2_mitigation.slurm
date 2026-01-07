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
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
WORKDIR="/home/hpc/b116ba/b117ba41/Masterarbeit/KonkreteVersuche/E3_drayson/model_collapse"
CONTAINER="$HOME/containers/python311.sif"

# Per-job output dir (stable, separated)
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"

# Use node-local temp if available (reduces /home usage)
TMPBASE="${SLURM_TMPDIR:-/tmp}/${USER}/model_collapse_${SLURM_JOB_ID}"
mkdir -p "$TMPBASE"

# Put heavy caches into TMP to reduce quota pressure (you are over /home soft quota)
export HF_HOME="$TMPBASE/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export PIP_CACHE_DIR="$TMPBASE/.pip_cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export WANDB_DISABLED=true

# Offline flags (keeps HF from trying network)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Python userbase inside project (persistent). If quota is tight, you can also move this to $TMPBASE,
# but then it reinstalls every job.
export PYTHONUSERBASE="$WORKDIR/.pyuserbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"

# ------------------------------------------------------------
# Enter repo
# ------------------------------------------------------------
if [ ! -d "$WORKDIR" ]; then
  echo "[ERROR] WORKDIR not found: $WORKDIR"
  exit 1
fi

cd "$WORKDIR"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] RUN_DIR: $RUN_DIR"
echo "[INFO] TMPBASE: $TMPBASE"

# ------------------------------------------------------------
# Dataset config + sanity check
# ------------------------------------------------------------
mkdir -p config/dataset
cat > config/dataset/wikitext2_local.yaml << 'EOF'
path: data/wikitext2
EOF
echo "[INFO] Wrote config/dataset/wikitext2_local.yaml"

for f in data/wikitext2/train.json data/wikitext2/validation.json data/wikitext2/test.json; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing dataset file: $f"
    echo "[HINT] If your repo stores data elsewhere, adapt config/dataset/wikitext2_local.yaml accordingly."
    exit 1
  fi
done
echo "[INFO] Human dataset JSONs found."

# ------------------------------------------------------------
# Model sanity check (offline requires local models)
# Put them in: $WORKDIR/_vendors/models/gpt2 and $WORKDIR/_vendors/models/modernbert_ai_detection
# ------------------------------------------------------------
GPT2_DIR="$WORKDIR/_vendors/models/gpt2"
DET_DIR="$WORKDIR/_vendors/models/modernbert_ai_detection"

for d in "$GPT2_DIR" "$DET_DIR"; do
  if [ ! -d "$d" ]; then
    echo "[ERROR] Missing local model dir: $d"
    echo "[HINT] You are running offline. Download on a machine with internet and copy into _vendors/models/."
    exit 1
  fi
  if [ ! -f "$d/config.json" ]; then
    echo "[ERROR] Local model dir missing config.json: $d"
    exit 1
  fi
done
echo "[INFO] Local model dirs look OK."

# ------------------------------------------------------------
# Install deps (idempotent)
# - Fixes certifi error
# - Avoids reinstall every run via a stamp file
# IMPORTANT: If you change deps, bump the stamp name to force re-install.
# ------------------------------------------------------------
STAMP="$PYTHONUSERBASE/.deps_ok_transformers_440_certifi_fix_v1"
if [ ! -f "$STAMP" ]; then
  echo "[INFO] Installing Python packages (forcing once via stamp) ..."
  singularity exec --nv "$CONTAINER" bash -lc "
    set -e
    cd '$WORKDIR'
    export PYTHONUSERBASE='$PYTHONUSERBASE'
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    export PIP_CACHE_DIR='$PIP_CACHE_DIR'
    export PIP_DISABLE_PIP_VERSION_CHECK=1

    echo '[DBG] python:' \$(which python)
    echo '[DBG] pip:' \$(which pip)
    python -V

    python -m pip install --upgrade --user pip setuptools wheel

    # ✅ critical fix: requests stack + certifi + hub
    python -m pip install --user --upgrade --force-reinstall \
      certifi requests urllib3 idna charset_normalizer huggingface_hub

    python - << 'PY'
import certifi
print('certifi OK:', certifi.__version__, 'at', certifi.__file__)
PY

    # Main deps (repo requirements often incomplete)
    python -m pip install --user \
      'torch>=2.0' \
      'transformers>=4.40' \
      'datasets>=2.14' \
      'accelerate>=0.20' \
      sentencepiece protobuf

    python -m pip install --user -r requirements.txt

    # Sanity check
    python - << 'PY'
import transformers, certifi, requests, huggingface_hub
print('transformers', transformers.__version__)
print('certifi', certifi.__version__)
print('requests', requests.__version__)
print('huggingface_hub', huggingface_hub.__version__)
PY
  "
  touch "$STAMP"
  echo "[INFO] Dependency stamp written: $STAMP"
else
  echo "[INFO] Deps already installed (stamp exists): $STAMP"
fi

# ------------------------------------------------------------
# Precreate expected generation folders
# ------------------------------------------------------------
NUM_ITERS=10
mkdir -p "$RUN_DIR"
for i in $(seq 0 $NUM_ITERS); do
  mkdir -p "$RUN_DIR/$i"
  cp -f "data/wikitext2/train.json" "$RUN_DIR/$i/data.json"
done
echo "[INFO] Precreated $((NUM_ITERS+1)) generation folders with data.json"

# ------------------------------------------------------------
# Run
# NOTE: We additionally pass explicit local model paths so HF never tries to fetch.
# If Hydra complains about unknown keys, we adjust the key names to match repo config.
# ------------------------------------------------------------
echo "[INFO] Starting main.py ..."
singularity exec --nv "$CONTAINER" bash -lc "
  set -e
  cd '$WORKDIR'

  export HF_HOME='$HF_HOME'
  export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
  export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
  export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'

  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
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
    wandb_disabled=true \
    model.model_name_or_path='$GPT2_DIR' \
    model.tokenizer_name='$GPT2_DIR' \
    detector.model_path='$DET_DIR' \
    detector.tokenizer_name='$DET_DIR'
"

echo "[INFO] Done at $(date)"
echo "[INFO] Outputs: $RUN_DIR"
