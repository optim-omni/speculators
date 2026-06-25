#!/bin/bash
# Online EAGLE1 training smoke for Qwen3-30B-A3B MoE on the CUDO vLLM 0.22 env.
#
# Defaults are intentionally tiny. Override variables from the shell for a larger
# run, for example:
#   MAX_SAMPLES=64 SEQ_LENGTH=1024 AUX_LAYER_ID=47 \
#     bash examples/train/eagle1_qwen30b_a3b_moe_online_smoke.sh

set -euo pipefail

# ============ Environment ============
WORKSPACE="${WORKSPACE:-/data00/kunlin/moe_qat_eagle1_vllm022_20260624}"
SPEC_PY="${SPEC_PY:-$WORKSPACE/.venv/bin/python}"
VLLM_PY="${VLLM_PY:-/data00/kunlin/moe_patch_envs/codex-vllm022-bf16-20260607/bin/python}"

# ============ Model/Data ============
MODEL="${MODEL:-/data00/kunlin/models/Qwen3-30B-A3B-merge-fused-vllm}"
DATASETS="${DATASETS:-magpie ultrachat}"
RUN_NAME="${RUN_NAME:-eagle1_qwen30b_a3b_online_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/runs/$RUN_NAME}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$OUTPUT_DIR/hidden_states}"
# Set MAX_SAMPLES=0 to process the full dataset split.
MAX_SAMPLES="${MAX_SAMPLES:-64}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-$SEQ_LENGTH}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((SEQ_LENGTH + 1))}"
MINIMUM_VALID_TOKENS="${MINIMUM_VALID_TOKENS:-1}"
OVERWRITE_DATA="${OVERWRITE_DATA:-0}"

# Qwen3-30B-A3B has 48 layers. EAGLE1 predicts the second-to-top-layer feature,
# so 47 is the default aux layer; the launch wrapper appends final layer 48 for
# verifier KL targets.
AUX_LAYER_ID="${AUX_LAYER_ID:-47}"
# Set DRAFT_VOCAB_SIZE=full to omit --draft-vocab-size and train/evaluate
# against the verifier's complete vocabulary without d2t/t2d remapping.
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-64000}"

# ============ Training ============
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
TTT_STEPS="${TTT_STEPS:-4}"
LOSS_FN="${LOSS_FN:-kl_div}"
EAGLE1_LOSS_MODE="${EAGLE1_LOSS_MODE:-eagle1_hass}"
EAGLE1_VLOSS_WEIGHT="${EAGLE1_VLOSS_WEIGHT:-1.0}"
EAGLE1_PLOSS_WEIGHT="${EAGLE1_PLOSS_WEIGHT:-0.1}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0}"
SAVE_BEST="${SAVE_BEST:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MOE_SPEC_EAGLE1_TORCH_COMPILE="${MOE_SPEC_EAGLE1_TORCH_COMPILE:-0}"

# ============ vLLM ============
VLLM_PORT="${VLLM_PORT:-8122}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
VLLM_ENABLE_EAGER="${VLLM_ENABLE_EAGER:-0}"
MOE_BACKEND="${MOE_BACKEND:-triton}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-240}"
MAX_RETRIES="${MAX_RETRIES:-3}"
ON_GENERATE="${ON_GENERATE:-delete}"

mkdir -p "$OUTPUT_DIR" "$HIDDEN_STATES_DIR"

DATA_ARGS=()
for dataset in $DATASETS; do
    DATA_ARGS+=(--data "$dataset")
done

PREPARE_ARGS=()
if [[ "$OVERWRITE_DATA" == "1" ]]; then
    PREPARE_ARGS+=(--overwrite)
fi
if [[ "$MAX_SAMPLES" != "0" ]]; then
    PREPARE_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

VLLM_EXTRA_ARGS=(
    --tensor-parallel-size "$VLLM_TP_SIZE"
    --port "$VLLM_PORT"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --moe-backend "$MOE_BACKEND"
)
if [[ "$VLLM_ENABLE_EAGER" == "1" ]]; then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
fi

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Step 1: Preparing Magpie + UltraChat data ==="
"$SPEC_PY" scripts/prepare_data.py \
    --model "$MODEL" \
    "${DATA_ARGS[@]}" \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH" \
    --minimum-valid-tokens "$MINIMUM_VALID_TOKENS" \
    --num-preprocessing-workers 4 \
    "${PREPARE_ARGS[@]}"

echo "=== Step 2: Launching vLLM hidden-state server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
env -u VLLM_GPUS -u VLLM_TP_SIZE -u VLLM_MAX_MODEL_LEN -u VLLM_ENABLE_EAGER \
    -u VLLM_USE_FLASHINFER_MOE_FP16 \
    "$VLLM_PY" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --target-layer-ids "$AUX_LAYER_ID" \
    -- "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM server on port ${VLLM_PORT}..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        wait "$VLLM_PID"
        exit 1
    fi
    sleep 2
done
echo "vLLM server ready."

TRAIN_CMD=(
    "$SPEC_PY" scripts/train.py
    --verifier-name-or-path "$MODEL"
    --data-path "$OUTPUT_DIR"
    --hidden-states-path "$HIDDEN_STATES_DIR"
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1"
    --save-path "$OUTPUT_DIR/checkpoints"
    --speculator-type eagle1_train
    --target-layer-ids "$AUX_LAYER_ID"
    --epochs "$EPOCHS"
    --lr "$LR"
    --total-seq-len "$TOTAL_SEQ_LEN"
    --ttt-steps "$TTT_STEPS"
    --loss-fn "$LOSS_FN"
    --eagle1-loss-mode "$EAGLE1_LOSS_MODE"
    --eagle1-vloss-weight "$EAGLE1_VLOSS_WEIGHT"
    --eagle1-ploss-weight "$EAGLE1_PLOSS_WEIGHT"
    --hidden-states-dtype bfloat16
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --log-freq 1
    --checkpoint-freq 1
    --scheduler-type cosine
    --on-missing generate
    --on-generate "$ON_GENERATE"
    --validation-split "$VALIDATION_SPLIT"
    --request-timeout "$REQUEST_TIMEOUT"
    --max-retries "$MAX_RETRIES"
)
if [[ -n "$DRAFT_VOCAB_SIZE" \
    && "$DRAFT_VOCAB_SIZE" != "full" \
    && "$DRAFT_VOCAB_SIZE" != "target" \
    && "$DRAFT_VOCAB_SIZE" != "none" \
    && "$DRAFT_VOCAB_SIZE" != "0" ]]; then
    TRAIN_CMD+=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
fi
if [[ "$SAVE_BEST" == "1" ]]; then
    TRAIN_CMD+=(--save-best)
fi

echo "=== Step 3: Training EAGLE1 drafter online ==="
if [[ "$NUM_TRAIN_GPUS" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    MOE_SPEC_EAGLE1_TORCH_COMPILE="$MOE_SPEC_EAGLE1_TORCH_COMPILE" \
        torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" "${TRAIN_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
    MOE_SPEC_EAGLE1_TORCH_COMPILE="$MOE_SPEC_EAGLE1_TORCH_COMPILE" \
        "${TRAIN_CMD[@]}"
fi

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
echo "Cached hidden states saved to $HIDDEN_STATES_DIR/"
