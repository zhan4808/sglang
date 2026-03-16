#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Nsight Compute (ncu) profiling for FlashInfer attention kernels in SGLang.
#
# This script runs ncu on the microbenchmark to capture detailed kernel metrics:
#   - SM utilization & occupancy
#   - Memory throughput & bandwidth
#   - Warp execution efficiency
#   - L1/L2 cache hit rates
#   - Register usage & spills
#
# Prerequisites:
#   - NVIDIA GPU with compute capability >= 8.0 (A100/H100/etc)
#   - Nsight Compute installed (comes with CUDA toolkit, or install standalone)
#   - Python environment with torch, flashinfer, sglang installed
#
# Usage:
#   chmod +x ncu_profile.sh
#   ./ncu_profile.sh                          # defaults: llama-8b, decode, bs=64, kv=2048
#   ./ncu_profile.sh deepseek-v2-lite decode 16 4096
#   ./ncu_profile.sh llama-8b prefill 4 2048
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${1:-llama-8b}"
MODE="${2:-decode}"
BATCH_SIZE="${3:-64}"
SEQ_OR_KV_LEN="${4:-2048}"
BACKEND="${5:-flashinfer}"
OUTPUT_DIR="${6:-./ncu_results}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/profile_attention_kernels.py"

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LABEL="${BACKEND}_${MODEL}_${MODE}_bs${BATCH_SIZE}_len${SEQ_OR_KV_LEN}"

echo "================================================================"
echo "NCU Profiling: ${LABEL}"
echo "================================================================"
echo "Backend:    $BACKEND"
echo "Model:      $MODEL"
echo "Mode:       $MODE"
echo "Batch size: $BATCH_SIZE"
echo "Seq/KV len: $SEQ_OR_KV_LEN"
echo "Output dir: $OUTPUT_DIR"
echo ""

# ── Step 1: Quick nsys overview to identify kernel names ────────────────────

echo "--- Step 1: nsys overview (identify top kernels) ---"
NSYS_OUT="${OUTPUT_DIR}/${LABEL}_${TIMESTAMP}_nsys"

if [ "$MODE" = "decode" ]; then
    LEN_ARG="--kv-lens ${SEQ_OR_KV_LEN}"
else
    LEN_ARG="--seq-lens ${SEQ_OR_KV_LEN}"
fi

nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --force-overwrite=true \
    -o "${NSYS_OUT}" \
    python "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --mode "$MODE" \
        --batch-sizes "$BATCH_SIZE" \
        $LEN_ARG \
        --backend "$BACKEND" \
        --ncu-mode

echo "nsys report saved to ${NSYS_OUT}.nsys-rep"

# Extract top GPU kernels by time
echo ""
echo "Top 15 GPU kernels by total time:"
nsys stats --report cuda_gpu_kern_sum "${NSYS_OUT}.nsys-rep" 2>/dev/null | head -20 || true
echo ""

# ── Step 2: NCU detailed kernel profiling ───────────────────────────────────

echo "--- Step 2: ncu detailed profiling ---"
NCU_OUT="${OUTPUT_DIR}/${LABEL}_${TIMESTAMP}"

# Full metrics collection: memory, compute, occupancy, warp efficiency
# --replay-mode kernel replays each kernel multiple times for accurate metrics
# --profile-from-start no  +  --nvtx  targets only the cudaProfiler region
ncu \
    --set full \
    --replay-mode kernel \
    --profile-from-start no \
    --target-processes all \
    --export "${NCU_OUT}" \
    --csv \
    --page raw \
    python "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --mode "$MODE" \
        --batch-sizes "$BATCH_SIZE" \
        $LEN_ARG \
        --backend "$BACKEND" \
        --ncu-mode \
    > "${NCU_OUT}_raw.csv" 2>&1

echo "NCU report saved to ${NCU_OUT}.ncu-rep"
echo "NCU CSV saved to ${NCU_OUT}_raw.csv"

# ── Step 3: NCU with specific metrics (lighter, faster) ────────────────────

echo ""
echo "--- Step 3: ncu targeted metrics ---"
NCU_LIGHT="${OUTPUT_DIR}/${LABEL}_${TIMESTAMP}_light"

ncu \
    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
lts__t_sectors_op_read_lookup_hit.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
smsp__inst_executed_pipe_tensor.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_tensor_pred_on.sum,\
launch__registers_per_thread,\
launch__occupancy,\
launch__block_size,\
launch__grid_size,\
gpu__time_duration.sum \
    --profile-from-start no \
    --target-processes all \
    --csv \
    python "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --mode "$MODE" \
        --batch-sizes "$BATCH_SIZE" \
        $LEN_ARG \
        --backend "$BACKEND" \
        --ncu-mode \
    > "${NCU_LIGHT}.csv" 2>&1

echo "NCU light CSV saved to ${NCU_LIGHT}.csv"

# ── Step 4: Run the analysis script ────────────────────────────────────────

echo ""
echo "--- Step 4: analysis ---"
if [ -f "${SCRIPT_DIR}/analyze_ncu.py" ]; then
    python "${SCRIPT_DIR}/analyze_ncu.py" \
        --csv "${NCU_LIGHT}.csv" \
        --label "$LABEL" \
        --output "${OUTPUT_DIR}/${LABEL}_${TIMESTAMP}_analysis.md"
fi

echo ""
echo "================================================================"
echo "Done! Files in ${OUTPUT_DIR}:"
ls -la "${OUTPUT_DIR}/${LABEL}"* 2>/dev/null || true
echo "================================================================"
