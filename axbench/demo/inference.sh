#!/usr/bin/env bash
set -uo pipefail

############################################
# Config
############################################

CONFIG="axbench/demo/sweep/hypersteer_supergpqa.yaml"
BASE_DUMP_DIR="runs/hypersteer-gemma2b-16000-2"
OUT_CSV="runs/all_reasoning_accuracies.csv"
STEERING_FACTORS="0,0.2,0.4,0.6,0.8,1"

mkdir -p "$(dirname "${OUT_CSV}")"

# CSV header
echo "benchmark,subset,mode,concept_prompt,steering_factor,accuracy" > "${OUT_CSV}"

############################################
# GPU + port management
############################################

GPUS=(5 6)
BASE_PORT=29502
gpu_idx=0
port_offset=0

next_gpu() {
  local gpu="${GPUS[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
  echo "$gpu"
}

next_port() {
  local port=$((BASE_PORT + port_offset))
  port_offset=$((port_offset + 1))
  echo "$port"
}

############################################
# Task definitions
############################################
# name | benchmark | discipline | field | concept_prompt

TASKS=(
  "gsm8k|gsm8k|||Basic Arithmetic Reasoning"

  "supergpqa_medicine|supergpqa|medicine||Epidemiology Reasoning"
  "supergpqa_physics|supergpqa||physics|Classical Mechanics Reasoning"
  "supergpqa_chemistry|supergpqa||chemistry|Organic Chemistry Reasoning"
  "supergpqa_history|supergpqa|history||Medieval European History Reasoning"
  "supergpqa_legal|supergpqa|legal||Constitutional Law Reasoning"
  "supergpqa_literature|supergpqa|Literature and Arts||Narrative Structure Reasoning"
)

############################################
# Runner
############################################

run_task () {
  IFS="|" read -r name benchmark discipline field concept_prompt <<< "$1"

  # GSM8K: prompt only
  if [[ "${benchmark}" == "gsm8k" ]]; then
    MODES=(prompt)
  else
    MODES=(auto prompt)
  fi

  for mode in "${MODES[@]}"; do
    gpu="$(next_gpu)"
    port="$(next_port)"

    echo "Running ${name} (${mode}) on GPU ${gpu}"

    CMD=(
      uv run python -m torch.distributed.run
      --nproc_per_node=1
      --rdzv_backend=c10d
      --rdzv_endpoint="localhost:${port}"
      axbench/scripts/inference.py
      --mode benchmark_steered
      --config "${CONFIG}"
      --dump_dir "${BASE_DUMP_DIR}"
      --steering_factors "${STEERING_FACTORS}"
      --benchmark "${benchmark}"
    )

    [[ -n "${discipline}" ]] && CMD+=(--supergpqa_discipline "${discipline}")
    [[ -n "${field}" ]] && CMD+=(--supergpqa_field "${field}")

    if [[ "${mode}" == "auto" ]]; then
      CMD+=(--supergpqa_auto_concept)
      MODE_LABEL="auto_concept"
      PROMPT_LABEL=""
    else
      CMD+=(--concept_prompt "${concept_prompt}")
      MODE_LABEL="prompt_concept"
      PROMPT_LABEL="${concept_prompt}"
    fi

    # Run inference
    OUTPUT="$(CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}" 2>&1 || true)"
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
      echo "⚠️  WARNING: inference command exited with status $STATUS for ${name} (${mode})"
    fi

    # ---- Extract per-factor accuracies from stdout ----
    echo "${OUTPUT}" | \
      grep -E "Benchmark:.*SUPERGPQA\+Steering.*Accuracy=" | \
      while read -r line; do
        FACTOR=$(echo "$line" | sed -E 's/.*factor=([0-9.]+).*/\1/')
        ACCURACY=$(echo "$line" | sed -E 's/.*Accuracy=([0-9.]+).*/\1/')
        echo "${benchmark},${discipline:-${field}},${MODE_LABEL},${PROMPT_LABEL},${FACTOR},${ACCURACY}" >> "${OUT_CSV}"
      done
  done
}

############################################
# Main loop
############################################

for task in "${TASKS[@]}"; do
  run_task "${task}"
done

echo "========================================"
echo "All runs completed."
echo "Accuracies saved to ${OUT_CSV}"
echo "========================================"