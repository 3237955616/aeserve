#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="/sgl-workspace/aeserve/benchmark/multi-model"
TASK_DIR="$BENCH_DIR/tasks"
LOG_DIR="$BENCH_DIR/run_logs"
mkdir -p "$LOG_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MODEL_NAME="${MODEL_NAME:-Llama-3.2-1B}"

TASKS=(
  "json_extract"
  "reasoning_gsm8k_style"
  "multi_chain_reasoning"
  "long_context_qa"
  "multi_turn_chat"
  "react_planning"
  "tip_suggestion"
)

cd "$BENCH_DIR"

for task in "${TASKS[@]}"; do
  exp_name="${MODEL_NAME}_${task}"
  echo "===== Running $exp_name ====="
  python3 benchmark.py --host "$HOST" --port "$PORT" --model-paths "$MODEL_NAME" --task-file "$TASK_DIR/${task}.jsonl" --micro-benchmark --num-gpus 1 --exp-name "$exp_name" | tee "$LOG_DIR/${exp_name}.log"
  sleep 3
done

echo "All tasks finished for model=$MODEL_NAME"