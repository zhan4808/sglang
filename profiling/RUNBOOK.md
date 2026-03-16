# Kernel-Level Architectural Analysis of FlashInfer Attention in SGLang

## What this is

A kernel-level profiling study of FlashInfer's attention kernels as used by SGLang on H100, with a FlashInfer vs Triton comparison. The goal is to produce data and analysis for a writeup titled something like:

**"FlashInfer vs Triton Attention on H100: Where the Cycles Actually Go"**

## What already exists (and what doesn't)

Framework-level benchmarks of SGLang/vLLM on H100/H200 exist — throughput, TTFT, tokens/sec. FlashInfer's own repo has kernel microbenchmarks. **What's rare in public**: Nsight Compute kernel-level analysis showing occupancy, bandwidth utilization, tensor core efficiency, cache behavior, and warp stalls. That's the gap this fills.

## Research questions

1. Is decode attention memory-bound on H100? At what % of peak HBM bandwidth (3.35 TB/s)?
2. Is prefill attention compute-bound? At what % of peak FP16 TFLOPS (990)?
3. How does FlashInfer compare to SGLang's Triton attention in bandwidth efficiency, register usage, and occupancy?
4. Does GQA's KV head sharing show up in L2 cache hit rates?
5. Does MLA's compressed KV actually reduce memory traffic vs GQA?
6. Where is the kernel leaving performance on the table, and what could be optimized?

## Hardware

H100 80GB SXM. Reference numbers:
- HBM bandwidth: 3.35 TB/s
- FP16 tensor core: 990 TFLOPS
- FP32: 67 TFLOPS
- L2 cache: 50 MB
- SMs: 132

## Setup

```bash
# 1. Clone and install
cd ~
git clone https://github.com/zhan4808/sglang.git
cd sglang
pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/  # adjust for your CUDA

# 2. Verify
python -c "import flashinfer; print(flashinfer.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd; print('Triton decode OK')"

# 3. Nsight tools
which ncu || apt-get install -y nsight-compute
which nsys || apt-get install -y nsight-systems-cli

# 4. For ncu, may need:
echo -1 > /proc/sys/kernel/perf_event_paranoid  # or run ncu with sudo
```

## Phase 1: Microbenchmark — FlashInfer vs Triton timing

This gives wall-clock timing, achieved bandwidth, and TFLOPS for both backends across model shapes.

```bash
cd ~/sglang/profiling

# ── Decode: the main comparison ──

# Llama-3-8B GQA: sweep batch sizes (both backends)
python profile_attention_kernels.py \
    --model llama-8b --mode decode --backend both \
    --batch-sizes 1,4,16,64,128,256 --kv-lens 2048 \
    --output results_decode_batchsweep.csv

# Llama-3-8B GQA: sweep KV lengths (both backends)
python profile_attention_kernels.py \
    --model llama-8b --mode decode --backend both \
    --batch-sizes 64 --kv-lens 256,512,1024,2048,4096,8192 \
    --output results_decode_kvsweep.csv

# DeepSeek MLA decode (FlashInfer only — no Triton MLA kernel)
python profile_attention_kernels.py \
    --model deepseek-v2-lite --mode decode --backend flashinfer \
    --batch-sizes 1,4,16,64,128,256 --kv-lens 2048 \
    --output results_mla_decode.csv

# ── Prefill: compute-bound regime ──

python profile_attention_kernels.py \
    --model llama-8b --mode prefill --backend both \
    --batch-sizes 1,4,16 --seq-lens 512,1024,2048,4096 \
    --output results_prefill.csv

# ── Full sweep for reference ──

python profile_attention_kernels.py \
    --model llama-8b --mode all --backend both \
    --output results_llama_full.csv
```

**What to look for in the CSV:**
- `bandwidth_GB_s`: decode should approach 2-3 TB/s at large batch sizes. If it's <1 TB/s, the kernel is underutilizing HBM.
- `tflops`: prefill should approach 500+ TFLOPS FP16 at long sequences. If it's <200, something is wrong.
- FlashInfer vs Triton: which is faster? At what batch sizes does the gap change?

## Phase 2: Nsight Compute — kernel-level deep dive

This is the core of the analysis. NCU tells you WHY a kernel is fast or slow.

### Key NCU metrics to collect

These are the metrics that classify compute-bound vs memory-bound:

```
# Memory subsystem
dram__throughput.avg.pct_of_peak_sustained_elapsed    # HBM bandwidth utilization
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum   # L1 hits
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum  # L1 misses
lts__t_sectors_op_read_lookup_hit.sum                       # L2 hits
lts__t_sectors_op_read_lookup_miss.sum                      # L2 misses

# Compute
sm__throughput.avg.pct_of_peak_sustained_elapsed      # SM utilization
smsp__inst_executed_pipe_tensor.sum                    # tensor core instructions
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum    # FMA ops (CUDA cores)
sm__warps_active.avg.pct_of_peak_sustained_active      # warp occupancy

# Launch config
launch__registers_per_thread      # register pressure
launch__occupancy                 # theoretical occupancy
launch__block_size
launch__grid_size

# Timing
gpu__time_duration.sum
```

### Run NCU profiling

```bash
cd ~/sglang/profiling

# FlashInfer decode — the key config
./ncu_profile.sh llama-8b decode 64 2048

# Triton decode — same config for comparison
# (The script profiles whatever backend profile_attention_kernels.py uses)
# For Triton-only profiling:
python profile_attention_kernels.py --model llama-8b --mode decode \
    --backend triton --batch-sizes 64 --kv-lens 2048 --ncu-mode

# Then run ncu on it:
ncu --set full --replay-mode kernel --profile-from-start no \
    --target-processes all --export ncu_results/triton_decode_bs64_kv2048 \
    python profile_attention_kernels.py --model llama-8b --mode decode \
        --backend triton --batch-sizes 64 --kv-lens 2048 --ncu-mode

# Small batch (latency regime)
./ncu_profile.sh llama-8b decode 1 2048

# Large batch (throughput regime)
./ncu_profile.sh llama-8b decode 256 2048

# Prefill
./ncu_profile.sh llama-8b prefill 4 2048
```

### What to look for in NCU output

| Metric | Decode expectation | Prefill expectation |
|--------|-------------------|-------------------|
| DRAM throughput | 60-90% of peak (memory-bound) | 20-40% of peak |
| SM throughput | 10-30% (underutilized) | 50-80% (compute-bound) |
| Tensor core instructions | Low (decode is memory-bound) | High (QK matmul + PV matmul) |
| Occupancy | Often 30-50% (register-limited) | 40-65% |
| L2 hit rate | High if GQA reuse works | Lower (streaming access) |
| Registers/thread | 128-255 (FlashInfer is register-heavy) | Similar |

**The money question**: Is FlashInfer's higher register usage buying it better bandwidth utilization than Triton? Or is Triton's simpler kernel achieving similar throughput with higher occupancy?

## Phase 3: End-to-end validation

Sanity-check that microbenchmark findings hold in real serving.

```bash
cd ~/sglang/profiling

# Torch profiler traces via bench_one_batch (no scheduler)
./profile_e2e.sh one-batch meta-llama/Llama-3.1-8B-Instruct ./e2e_results

# nsys system trace (kernel timeline)
./profile_e2e.sh nsys-one-batch meta-llama/Llama-3.1-8B-Instruct ./e2e_results
```

Open traces in https://ui.perfetto.dev/ — look for attention kernel duration as a fraction of total step time.

## Phase 4: Collect results and draft writeup

### File inventory

After all runs, you should have:
```
results_decode_batchsweep.csv     # FlashInfer vs Triton decode, batch size sweep
results_decode_kvsweep.csv        # FlashInfer vs Triton decode, KV length sweep
results_mla_decode.csv            # MLA decode (FlashInfer only)
results_prefill.csv               # FlashInfer vs Triton prefill
ncu_results/*_analysis.md         # Auto-generated bottleneck reports
ncu_results/*.ncu-rep             # NCU reports (open in NCU GUI)
e2e_results/*.json                # Torch profiler traces (open in Perfetto)
```

### Writeup structure

**Title**: "FlashInfer vs Triton Attention on H100: A Kernel-Level Analysis"

1. **Setup**: GPU specs, software versions, model shapes tested
2. **Decode analysis**:
   - Chart: batch_size vs achieved bandwidth (GB/s), with 3.35 TB/s peak line
   - FlashInfer vs Triton speedup ratio across batch sizes
   - NCU breakdown: classify as memory-bound, show % of peak DRAM
   - Register usage comparison (FlashInfer vs Triton)
   - L2 cache hit rates (does GQA head sharing help?)
3. **Prefill analysis**:
   - Chart: seq_len vs TFLOPS, with 990 TFLOPS peak line
   - Tensor core utilization from NCU
4. **MLA vs GQA**: Does MLA's compressed KV reduce memory traffic?
5. **Key findings**: 2-3 specific bottlenecks with data
6. **Optimization ideas**: Even a small suggestion makes the post stronger

### Example findings to look for

- "FlashInfer decode achieves 2.1 TB/s (63% of peak) at bs=256, while Triton achieves 1.8 TB/s (54%) — the gap comes from FlashInfer's use of TMA loads on Hopper"
- "Occupancy is 38% for FlashInfer (192 regs/thread) vs 52% for Triton (96 regs/thread), but FlashInfer compensates with better memory access patterns"
- "L2 cache hit rate is 72% for Llama-8B GQA (4:1 Q:KV ratio) but only 45% for DeepSeek MLA (1:1 ratio), confirming that GQA's head sharing provides meaningful cache reuse"
- "Prefill attention achieves 680 TFLOPS (69% of peak) — the gap is primarily warp divergence from causal masking"

## Troubleshooting

- **flashinfer import fails**: Check CUDA version matches. `pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/`
- **BatchMLAPagedAttentionWrapper not found**: Need flashinfer >= 0.2.0. Skip MLA if unavailable.
- **ncu permission denied**: Run `echo -1 > /proc/sys/kernel/perf_event_paranoid` or use `sudo ncu ...`
- **ncu very slow**: Expected — it replays kernels many times. `--ncu-mode` limits to 5 iterations.
- **OOM**: Reduce batch sizes. Decode allocates O(bs * kv_len * heads * dim) for KV cache.
- **Triton compilation slow on first run**: Normal. Subsequent runs use cached kernels.
