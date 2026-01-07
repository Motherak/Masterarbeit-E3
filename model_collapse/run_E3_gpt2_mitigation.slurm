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

WORKDIR="/home/hpc/b116ba/b117ba41/Masterarbeit/KonkreteVersuche/E3_drayson/model_collapse"
CONTAINER="$HOME/containers/python311.sif"
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"

TMPBASE="${SLURM_TMPDIR:-/tmp}/${USER}/model_collapse_${SLURM_JOB_ID}"
mkdir -p "$TMPBASE"

export HF_HOME="$TMPBASE/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export PIP_CACHE_DIR="$TMPBASE/.pip_cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export WANDB_DISABLED=true

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTHONUSERBASE="$WORKDIR/.pyuserbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"
unset PYTHONNOUSERSITE
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:${PYTHONPATH:-}"

cd "$WORKDIR"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] RUN_DIR: $RUN_DIR"
echo "[INFO] TMPBASE: $TMPBASE"

# local models
GPT2_DIR="$WORKDIR/_vendors/models/gpt2"
DET_DIR="$WORKDIR/_vendors/models/modernbert_ai_detection"
WHEEL_DIR="$WORKDIR/_vendors/wheels"

for d in "$GPT2_DIR" "$DET_DIR"; do
  [ -d "$d" ] || { echo "[ERROR] Missing local model dir: $d"; exit 1; }
  [ -f "$d/config.json" ] || { echo "[ERROR] Missing config.json in: $d"; exit 1; }
done
ls "$WHEEL_DIR"/certifi-*.whl >/dev/null 2>&1 || { echo "[ERROR] certifi wheel missing in $WHEEL_DIR"; exit 1; }

# dataset cfg
mkdir -p config/dataset
cat > config/dataset/wikitext2_local.yaml << 'EOF'
path: data/wikitext2
EOF

for f in data/wikitext2/train.json data/wikitext2/validation.json data/wikitext2/test.json; do
  [ -f "$f" ] || { echo "[ERROR] Missing dataset file: $f"; exit 1; }
done

# install deps (force once)
STAMP="$PYTHONUSERBASE/.deps_ok_transformers_440_offline_certifi_target_v3"
if [ ! -f "$STAMP" ]; then
  echo "[INFO] Installing Python packages (offline certifi --target) ..."
  singularity exec --nv "$CONTAINER" bash -lc "
    set -e
    cd '$WORKDIR'

    export PYTHONUSERBASE='$PYTHONUSERBASE'
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    unset PYTHONNOUSERSITE
    export PYTHONPATH='$PYTHONUSERBASE/lib/python3.11/site-packages':\${PYTHONPATH:-}
    export PIP_CACHE_DIR='$PIP_CACHE_DIR'
    export PIP_DISABLE_PIP_VERSION_CHECK=1

    echo '[DBG] python:' \$(which python)
    python -V
    echo '[DBG] PYTHONUSERBASE:' \$PYTHONUSERBASE
    echo '[DBG] PYTHONPATH:' \$PYTHONPATH

    python -m pip install --upgrade --user pip setuptools wheel

    # ✅ certifi exakt in userbase site-packages installieren
    python -m pip install --no-index --find-links '$WHEEL_DIR' \
      --upgrade --force-reinstall \
      --target '$PYTHONUSERBASE/lib/python3.11/site-packages' \
      certifi

    python - << 'PY'
import certifi
print('certifi OK:', certifi.__version__, 'at', certifi.__file__)
PY

    # übrige deps
    python -m pip install --user \
      'torch>=2.0' \
      'transformers>=4.40' \
      'datasets>=2.14' \
      'accelerate>=0.20' \
      sentencepiece protobuf

    python -m pip install --user -r requirements.txt

    python - << 'PY'
import transformers, certifi
print('transformers', transformers.__version__)
print('certifi', certifi.__version__)
PY
  "
  touch "$STAMP"
  echo "[INFO] Dependency stamp written: $STAMP"
else
  echo "[INFO] Deps already installed (stamp exists): $STAMP"
fi

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
  unset PYTHONNOUSERSITE
  export PYTHONPATH='$PYTHONUSERBASE/lib/python3.11/site-packages':\${PYTHONPATH:-}

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
