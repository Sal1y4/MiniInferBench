#!/usr/bin/env bash
# Cross-check `mib vllm` E2E timing against vLLM's own `vllm bench latency`.
#
# Runs `vllm bench latency` with the same engine settings that `mib vllm`
# uses for the published grid (docs/VLLM.md, "Cross-check"), one engine per
# point, and writes one JSON file per point. Compare each file's mean of
# `latencies` with `summary.e2e_wall_ms_mean` of the matching record.
#
# Usage: bash scripts/vllm_bench_crosscheck.sh [OUT_DIR]   (from the repo root)
set -u

out=${1:-results/vllm-bench-latency}
mkdir -p "$out"

# Same as the backend sets before importing vLLM (docs/VLLM.md, "Setup").
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# Engine sized like the published sweep: max_num_seqs = largest batch (8),
# max_model_len = longest input + output (1024 + 128), and a token budget
# that holds the whole batch's prefill in one step (8 * 1152).
common=(
  --model HuggingFaceTB/SmolLM2-360M-Instruct --dtype bfloat16
  --gpu-memory-utilization 0.4 --max-model-len 1152 --max-num-seqs 8
  --max-num-batched-tokens 9216 --seed 0 --disable-detokenize
  --output-len 128 --num-iters-warmup 6 --num-iters 20
)

run() {  # mode batch input [extra vllm args...]
  local mode=$1 bs=$2 in=$3
  shift 3
  local name="$out/${mode}_bs${bs}_in${in}"
  if [ -f "$name.json" ]; then
    echo "skip $name.json"
    return
  fi
  echo "=== $mode bs=$bs in=$in load=$(cut -d' ' -f1 /proc/loadavg)"
  if .venv-vllm/bin/vllm bench latency "${common[@]}" --batch-size "$bs" \
      --input-len "$in" --output-json "$name.json" "$@" > "$name.log" 2>&1; then
    grep "Avg latency" "$name.log"
  else
    echo "FAILED, see $name.log"
  fi
}

for bs in 1 2 4 8; do
  for in in 128 512 1024; do
    run graph "$bs" "$in"
  done
done
for bs in 1 8; do
  for in in 128 1024; do
    run eager "$bs" "$in" --enforce-eager
  done
done
