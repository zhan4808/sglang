#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end SGLang server profiling using built-in tools.
#
# This complements the microbenchmark by showing how attention kernels
# behave in a realistic serving context with the full scheduler and
# memory management stack.
#
# Usage:
#   ./profile_e2e.sh                  # defaults: Llama-3.1-8B, bench_one_batch
#   ./profile_e2e.sh server           # full server + bench_serving
#   ./profile_e2e.sh one-batch        # bench_one_batch (kernel-level)
#   ./profile_e2e.sh nsys-server      # nsys profile of full server
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODE="${1:-one-batch}"
MODEL="${2:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${3:-./e2e_results}"

mkdir -p "$OUTPUT_DIR"
export SGLANG_TORCH_PROFILER_DIR="$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "E2E Profiling: mode=$MODE  model=$MODEL"
echo "================================================================"

case "$MODE" in
    one-batch)
        # ── Kernel-level profiling via bench_one_batch ──
        # This bypasses scheduler, directly calls ModelRunner
        # Good for isolating attention kernel behavior

        echo "--- Profiling decode (batch=64, input=512, output=32) ---"
        python -m sglang.bench_one_batch \
            --model-path "$MODEL" \
            --batch-size 64 \
            --input-len 512 \
            --output-len 32 \
            --profile \
            --load-format dummy

        echo ""
        echo "--- Profiling decode (batch=1, input=4096, output=32) ---"
        python -m sglang.bench_one_batch \
            --model-path "$MODEL" \
            --batch-size 1 \
            --input-len 4096 \
            --output-len 32 \
            --profile \
            --load-format dummy

        echo ""
        echo "--- Profiling decode (batch=256, input=256, output=32) ---"
        python -m sglang.bench_one_batch \
            --model-path "$MODEL" \
            --batch-size 256 \
            --input-len 256 \
            --output-len 32 \
            --profile \
            --load-format dummy

        # Also try with reduced layers for faster iteration
        echo ""
        echo "--- Profiling with 1 layer (fast, for attention kernel isolation) ---"
        python -m sglang.bench_one_batch \
            --model-path "$MODEL" \
            --batch-size 64 \
            --input-len 1024 \
            --output-len 10 \
            --profile \
            --load-format dummy \
            --json-model-override-args '{"num_hidden_layers": 1}'
        ;;

    server)
        # ── Full server profiling via bench_serving ──
        echo "Starting SGLang server with dummy weights..."
        python -m sglang.launch_server \
            --model-path "$MODEL" \
            --load-format dummy \
            --port 30000 &
        SERVER_PID=$!
        trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

        # Wait for server to be ready
        echo "Waiting for server..."
        for i in $(seq 1 120); do
            if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
                echo "Server ready!"
                break
            fi
            sleep 1
        done

        echo "--- Running bench_serving with profiling ---"
        python -m sglang.bench_serving \
            --backend sglang \
            --model "$MODEL" \
            --num-prompts 10 \
            --sharegpt-output-len 100 \
            --profile

        echo "Traces saved to $OUTPUT_DIR"
        ;;

    nsys-server)
        # ── nsys profiling of full server ──
        NSYS_OUT="${OUTPUT_DIR}/sglang_e2e_${TIMESTAMP}"

        echo "Starting SGLang server under nsys..."
        nsys profile \
            --trace-fork-before-exec=true \
            --cuda-graph-trace=node \
            -o "$NSYS_OUT" \
            --duration 120 \
            python -m sglang.launch_server \
                --model-path "$MODEL" \
                --load-format dummy \
                --disable-radix-cache \
                --port 30000 &
        NSYS_PID=$!
        trap "kill $NSYS_PID 2>/dev/null; wait $NSYS_PID 2>/dev/null" EXIT

        echo "Waiting for server..."
        for i in $(seq 1 120); do
            if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
                echo "Server ready!"
                break
            fi
            sleep 1
        done

        echo "--- Running bench_serving ---"
        python -m sglang.bench_serving \
            --backend sglang \
            --model "$MODEL" \
            --num-prompts 50 \
            --dataset-name random \
            --random-input 512 \
            --random-output 128

        # Stop nsys
        echo "Stopping nsys..."
        nsys sessions list 2>/dev/null | grep profile | awk '{print $1}' | while read sid; do
            nsys stop --session="$sid" 2>/dev/null || true
        done

        echo "nsys report saved to ${NSYS_OUT}.nsys-rep"
        ;;

    nsys-one-batch)
        # ── nsys profiling of single batch (cleanest kernel traces) ──
        NSYS_OUT="${OUTPUT_DIR}/sglang_onebatch_${TIMESTAMP}"

        echo "--- nsys profiling bench_one_batch ---"
        nsys profile \
            --trace-fork-before-exec=true \
            --cuda-graph-trace=node \
            -o "$NSYS_OUT" \
            python -m sglang.bench_one_batch \
                --model-path "$MODEL" \
                --batch-size 64 \
                --input-len 512 \
                --output-len 32 \
                --load-format dummy

        echo "nsys report saved to ${NSYS_OUT}.nsys-rep"

        echo ""
        echo "Top kernels:"
        nsys stats --report cuda_gpu_kern_sum "${NSYS_OUT}.nsys-rep" 2>/dev/null | head -20 || true
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {one-batch|server|nsys-server|nsys-one-batch} [model] [output_dir]"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Done! Traces in: $OUTPUT_DIR"
echo "View in: https://ui.perfetto.dev/ or chrome://tracing"
echo "================================================================"
