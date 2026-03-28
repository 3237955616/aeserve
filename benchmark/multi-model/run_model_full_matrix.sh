#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="/sgl-workspace/aeserve/benchmark/multi-model"
TASK_DIR="$BENCH_DIR/tasks_v2"

MODEL_NAME="${MODEL_NAME:-Llama-3.2-1B-Instruct}"
MODEL_FAMILY="${MODEL_FAMILY:-Llama}"
PARAMS_B="${PARAMS_B:-3.2}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-131072}"
QUANTIZATION="${QUANTIZATION:-none}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
REPEATS="${REPEATS:-2}"
RATES_SMALL="${RATES_SMALL:-1 2 4 8}"
RATES_LARGE="${RATES_LARGE:-1 2 4}"
MODEL_SIZE_TIER="${MODEL_SIZE_TIER:-small}"
BENCH_MODE="${BENCH_MODE:-micro}"
NOTES="${NOTES:-}"

if [[ "$MODEL_SIZE_TIER" == "large" ]]; then
  RATES="$RATES_LARGE"
else
  RATES="$RATES_SMALL"
fi

RUN_ROOT="$BENCH_DIR/results_archive/$MODEL_NAME"
RESULTS_PATH="$RUN_ROOT/benchmark-results"
REQUEST_OUT="$RUN_ROOT/output-requests"
RUN_LOGS="$RUN_ROOT/run_logs"
GPU_LOGS="$RUN_ROOT/gpu_logs"
TMP_DIR="$RUN_ROOT/tmp_taskfiles"
META_DIR="$RUN_ROOT/meta"
SKIP_LOG="$META_DIR/skipped_runs.log"

mkdir -p "$RESULTS_PATH" "$REQUEST_OUT" "$RUN_LOGS" "$GPU_LOGS" "$TMP_DIR" "$META_DIR"

TASK_FILES=(
  "json_extract__short.jsonl"
  "json_extract__base.jsonl"
  "json_extract__long.jsonl"
  "reasoning_gsm8k_style__short.jsonl"
  "reasoning_gsm8k_style__base.jsonl"
  "reasoning_gsm8k_style__long.jsonl"
  "multi_chain_reasoning__short.jsonl"
  "multi_chain_reasoning__base.jsonl"
  "multi_chain_reasoning__long.jsonl"
  "long_context_qa__short.jsonl"
  "long_context_qa__base.jsonl"
  "long_context_qa__long.jsonl"
  "line_retrieval__short.jsonl"
  "line_retrieval__base.jsonl"
  "line_retrieval__long.jsonl"
  "multi_turn_chat__short.jsonl"
  "multi_turn_chat__base.jsonl"
  "multi_turn_chat__long.jsonl"
  "react_planning__short.jsonl"
  "react_planning__base.jsonl"
  "react_planning__long.jsonl"
  "tip_suggestion__short.jsonl"
  "tip_suggestion__base.jsonl"
  "tip_suggestion__long.jsonl"
)

cd "$BENCH_DIR"

python3 - <<PY
import csv, os
p = os.path.join("$META_DIR", "model_registry.csv")
exists = os.path.exists(p)
with open(p, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if not exists:
        w.writerow(["model_name","model_family","params_b","context_window","quantization","dtype","tp_size","num_gpus","notes"])
    w.writerow(["$MODEL_NAME","$MODEL_FAMILY","$PARAMS_B","$CONTEXT_WINDOW","$QUANTIZATION","$DTYPE","$TP_SIZE","$NUM_GPUS","$NOTES"])
print("wrote", p)
PY

cp "$TASK_DIR/task_manifest.jsonl" "$META_DIR/task_manifest.jsonl" || true
git -C /sgl-workspace/aeserve rev-parse HEAD > "$META_DIR/git_commit.txt" 2>/dev/null || true
nvidia-smi -L > "$META_DIR/nvidia_smi_L.txt" 2>/dev/null || true
nvidia-smi topo -m > "$META_DIR/nvidia_smi_topo.txt" 2>/dev/null || true
nvidia-smi -q > "$META_DIR/nvidia_smi_q.txt" 2>/dev/null || true

for task_file in "${TASK_FILES[@]}"; do
  base_name="${task_file%.jsonl}"
  for rate in $RATES; do
    for rep in $(seq 1 "$REPEATS"); do
      if [[ -f "$META_DIR/run_registry.csv" ]] && grep -Fq ",$MODEL_NAME,$base_name,$rate,$rep,$BENCH_MODE," "$META_DIR/run_registry.csv"; then
        echo "Skipping completed run: model=$MODEL_NAME task=$base_name rate=$rate rep=$rep mode=$BENCH_MODE"
        continue
      fi

      temp_task="$TMP_DIR/${base_name}__r${rate}__rep${rep}.jsonl"

      python3 - <<PY
import json
from transformers import AutoTokenizer

src = "$TASK_DIR/$task_file"
dst = "$temp_task"
rate = float("$rate")
context_window = int("$CONTEXT_WINDOW")
model_name = "$MODEL_NAME"
reserve_tokens = 256

tokenizer_path = f"/data/zjy/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=True,
    local_files_only=True,
)

kept = 0
skipped = 0

with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        obj = json.loads(line)

        prompt = obj["prompt"]
        output_len = int(obj.get("output_len", 0))

        prompt_len_real = len(tokenizer.encode(prompt, add_special_tokens=True))
        obj["prompt_len"] = prompt_len_real

        if prompt_len_real + output_len + reserve_tokens > context_window:
            skipped += 1
            continue

        obj["arrival_time"] = kept / rate
        fout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
        kept += 1

with open("$SKIP_LOG", "a", encoding="utf-8") as f:
    f.write(
        json.dumps({
            "model_name": model_name,
            "tokenizer_path": tokenizer_path,
            "task_file": "$task_file",
            "base_name": "$base_name",
            "rate": "$rate",
            "repeat_id": "$rep",
            "context_window": context_window,
            "reserve_tokens": reserve_tokens,
            "kept": kept,
            "skipped": skipped,
            "all_skipped": kept == 0
        }, ensure_ascii=False) + "\\n"
    )

print(f"wrote {dst}, kept={kept}, skipped={skipped}, tokenizer_path={tokenizer_path}")
PY

      if [[ ! -s "$temp_task" ]]; then
        echo "Skipping $task_file at rate=$rate rep=$rep because no valid samples fit CONTEXT_WINDOW=$CONTEXT_WINDOW" | tee -a "$SKIP_LOG"
        continue
      fi

      exp_name="${MODEL_NAME}__${base_name}__r${rate}__rep${rep}"
      run_log="$RUN_LOGS/${exp_name}.log"
      gpu_log="$GPU_LOGS/${exp_name}.csv"

      echo "===== Running $exp_name ====="

      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu --format=csv,noheader,nounits -l 1 > "$gpu_log" &
      GPU_LOGGER_PID=$!

      if [[ "$BENCH_MODE" == "e2e" ]]; then
        python3 benchmark.py --host "$HOST" --port "$PORT" --model-paths "$MODEL_NAME" --task-file "$temp_task" --e2e-benchmark --num-gpus "$NUM_GPUS" --exp-name "$exp_name" --results-path "$RESULTS_PATH" --request-path "$REQUEST_OUT" | tee "$run_log"
      else
        python3 benchmark.py --host "$HOST" --port "$PORT" --model-paths "$MODEL_NAME" --task-file "$temp_task" --micro-benchmark --num-gpus "$NUM_GPUS" --exp-name "$exp_name" --results-path "$RESULTS_PATH" --request-path "$REQUEST_OUT" | tee "$run_log"
      fi

      kill "$GPU_LOGGER_PID" 2>/dev/null || true
      wait "$GPU_LOGGER_PID" 2>/dev/null || true

      python3 - <<PY
import csv, os, time
p = os.path.join("$META_DIR", "run_registry.csv")
exists = os.path.exists(p)
with open(p, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if not exists:
        w.writerow(["timestamp","model_name","task_file","rate","repeat_id","benchmark_mode","results_path","request_out","run_log","gpu_log"])
    w.writerow([int(time.time()),"$MODEL_NAME","$base_name","$rate","$rep","$BENCH_MODE","$RESULTS_PATH","$REQUEST_OUT","$run_log","$gpu_log"])
print("updated", p)
PY

      sleep 3
    done
  done
done

echo "All runs finished for $MODEL_NAME"