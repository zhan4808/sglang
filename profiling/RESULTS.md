# FlashInfer vs Triton Attention on H100: Where the Cycles Actually Go

## Setup

| Component | Version |
|-----------|---------|
| GPU | NVIDIA H100 80GB HBM3 (SXM) |
| HBM bandwidth | 3.35 TB/s peak |
| FP16 tensor core | 990 TFLOPS peak |
| SMs | 132 |
| L2 cache | 50 MB |
| CUDA | 12.8 |
| PyTorch | 2.9.1+cu128 |
| FlashInfer | 0.6.6 |
| Triton | 3.5.1 |
| Model shapes | Llama-3-8B (32 Q heads, 8 KV heads, dim=128, GQA 4:1) |
| GPU clocks | Not locked (boost under load); memory at max 2619 MHz |

**Note on methodology**: All timing uses CUDA events with median of 100+ iterations after 10+ warmup runs. Reproducibility verified across 3 independent trials — results are stable to <1% CV at bs>=64 (see Appendix A). GPU clocks were not locked via `nvidia-smi -lgc`; for publication-quality results, locking is recommended.

---

## 1. Decode Attention: Memory-Bound Regime

### 1.1 Batch Size Sweep (kv_len=2048)

| Batch Size | FlashInfer (ms) | Triton (ms) | FlashInfer BW (GB/s) | Triton BW (GB/s) | Speedup |
|-----------|----------------|-------------|---------------------|-----------------|---------|
| 1 | 0.028 | 0.057 | 298 | 147 | 2.03x |
| 4 | 0.032 | 0.058 | 1,056 | 582 | 1.82x |
| 16 | 0.058 | 0.070 | 2,317 | 1,910 | 1.21x |
| 64 | 0.187 | 0.218 | 2,870 | 2,470 | 1.16x |
| 128 | 0.364 | 0.416 | 2,957 | 2,585 | 1.14x |
| 256 | 0.720 | 0.806 | 2,987 | 2,669 | 1.12x |

**Bandwidth formula**: `BW = (KV_read + Q_read + O_write) / median_time`, where `KV_read = 2 * bs * kv_len * num_kv_heads * head_dim * 2 bytes`. KV reads account for 99.8% of total bytes; Q and O are negligible.

**Key finding**: FlashInfer is consistently faster, but the gap narrows at larger batch sizes (2x at bs=1 -> 1.12x at bs=256). At bs=256, FlashInfer achieves **2,987 GB/s (89% of peak HBM bandwidth)**, while Triton achieves **2,669 GB/s (80%)**. Both are clearly memory-bound.

### 1.2 KV Length Sweep (batch_size=64)

| KV Length | FlashInfer (ms) | Triton (ms) | FlashInfer BW (GB/s) | Triton BW (GB/s) | Speedup |
|----------|----------------|-------------|---------------------|-----------------|---------|
| 256 | 0.037 | 0.065 | 1,859 | 1,053 | 1.76x |
| 512 | 0.059 | 0.069 | 2,273 | 1,950 | 1.17x |
| 1024 | 0.102 | 0.113 | 2,648 | 2,389 | 1.11x |
| 2048 | 0.187 | 0.217 | 2,875 | 2,476 | 1.16x |
| 4096 | 0.360 | 0.409 | 2,984 | 2,624 | 1.14x |
| 8192 | 0.716 | 0.790 | 3,000 | 2,720 | 1.10x |

**Key finding**: FlashInfer peaks at **3,000 GB/s (89.6% of peak)** at kv_len=8192. The gap is widest at short KV lengths (1.76x at kv=256) where launch overhead and kernel scheduling dominate. At longer KV lengths, both converge toward peak bandwidth.

### 1.3 NCU Kernel-Level Analysis (bs=64, kv=2048)

#### FlashInfer Decode

| Metric | Value |
|--------|-------|
| **Kernel** | `BatchPrefillWithPagedKVCacheKernel` (single fused kernel) |
| **Classification** | MEMORY-BOUND |
| **DRAM throughput** | 84% of peak |
| **SM throughput** | 17% of peak |
| **Registers/thread** | **183** |
| **Active warps** | 12.2% of peak |
| **Block size** | (32, 1, 4) = 128 threads |
| **Grid size** | (64, 1, 8) = 512 blocks |
| **L1 global load hit rate** | 1.0% (162K hit / 16.9M miss sectors) |
| **L2 hit rate** | 0.4% (112K hit / 25.3M miss sectors) |
| **Tensor core insts** | 2,162,688 |
| **Local memory (spills)** | n/a (not reported by NCU — see note) |
| **Duration** | ~191 us |

#### Triton Decode

| Metric | Stage 1 (`_fwd_grouped_kernel_stage1`) | Stage 2 (`_fwd_kernel_stage2`) |
|--------|---------------------------------------|-------------------------------|
| **Classification** | MEMORY-BOUND | COMPUTE-BOUND |
| **DRAM throughput** | 76% of peak | 25% of peak |
| **SM throughput** | 28% of peak | 37% of peak |
| **Registers/thread** | **76** | **30** |
| **Active warps** | 35.4% of peak | 73.8% of peak |
| **Block size** | 128 | 128 |
| **Grid size** | (64, 8, 8) = 4096 | (64, 32, 1) = 2048 |
| **L1 global load hit rate** | 4.7% (839K hit / 17.2M miss) | 33.8% (138K hit / 271K miss) |
| **L2 hit rate** | 1.0% (266K hit / 25.4M miss) | 8.9% (39K hit / 400K miss) |
| **Tensor core insts** | 2,097,152 | 0 |
| **Local memory (spills)** | n/a (not reported by NCU — see note) | n/a |
| **Duration** | ~217 us | ~10 us |

**Note on spill measurement**: NCU returned `n/a` for `l1tex__data_pipe_lsu_wavefronts_mem_local_op_{ld,st}.sum` for ALL kernels (both FlashInfer and Triton, both decode and prefill). This is a known issue with Hopper (sm_90) — the local memory wavefront counters are not always populated. **We cannot confirm or deny register spilling from the available NCU data.** The 183 vs 76 register difference is confirmed, and its impact on occupancy (12.2% vs 35.4% active warps) is measured, but whether FlashInfer or Triton actually spills to local memory remains unverified.

### 1.4 Architectural Analysis

**The register tradeoff**: FlashInfer uses **183 registers/thread** (2.4x more than Triton's 76). This severely limits occupancy (12.2% active warps vs Triton's 35.4%) but FlashInfer compensates with **better memory access patterns** — achieving 84% DRAM throughput vs Triton's 76%.

**Single kernel vs two-phase**: FlashInfer fuses everything into one kernel. Triton uses a split-KV approach with Stage 1 (memory-bound scatter-gather over KV splits) + Stage 2 (compute-bound softmax reduction). Stage 2 adds ~10us overhead (4.4% of total Triton time) but achieves excellent 73.8% warp occupancy with only 30 registers.

**Cache behavior**: Both have near-zero L2 hit rates on the main kernel (<1%), confirming decode attention is dominated by streaming KV cache reads from HBM. GQA's 4:1 head sharing doesn't provide meaningful L2 reuse at this problem size — the `64 * 2048 * 8 * 128 * 2 = 256 MB` KV cache far exceeds the 50 MB L2.

**Tensor core utilization**: Both use similar tensor core instruction counts (~2.1M). Neither backend spends significant time on CUDA core FP — tensor cores dominate the compute.

---

## 2. Prefill Attention: Compute-Bound Regime

### 2.1 Timing Results

**FLOPS formula note**: The TFLOPS values below use the formula `2 * B * (S²/2) * H * d`, which counts only the QK^T matmul (with causal masking). The standard convention counts both QK^T and AV matmuls: `4 * B * (S²/2) * H * d = 2x higher`. The table below reports the **script's values (QK^T only)** alongside **corrected values (both matmuls)**.

| Config | FlashInfer (ms) | Triton (ms) | FI TFLOPS (QK only) | FI TFLOPS (corrected) | Tri TFLOPS (corrected) | Speedup |
|--------|----------------|-------------|---------------------|----------------------|----------------------|---------|
| bs=1, seq=512 | 0.024 | 0.040 | 44 | 88 | 54 | 1.63x |
| bs=1, seq=1024 | 0.033 | 0.069 | 132 | 264 | 124 | 2.12x |
| bs=1, seq=2048 | 0.089 | 0.214 | 194 | 387 | 160 | 2.41x |
| bs=1, seq=4096 | 0.283 | 0.730 | 243 | 485 | 188 | 2.58x |
| bs=4, seq=512 | 0.039 | 0.070 | 109 | 218 | 124 | 1.77x |
| bs=4, seq=1024 | 0.098 | 0.212 | 175 | 349 | 162 | 2.15x |
| bs=4, seq=2048 | 0.307 | 0.753 | 224 | **448** | 183 | 2.45x |
| bs=4, seq=4096 | 0.995 | 2.698 | 276 | **552** | 204 | 2.71x |
| bs=16, seq=512 | 0.141 | 0.252 | 122 | 244 | 137 | 1.79x |
| bs=16, seq=1024 | 0.372 | 0.830 | 185 | 369 | 166 | 2.23x |
| bs=16, seq=2048 | 1.238 | 2.822 | 222 | 444 | 195 | 2.28x |
| bs=16, seq=4096 | 4.299 | 10.531 | 256 | **512** | 209 | 2.45x |

**Key finding**: FlashInfer dominates prefill — **2-2.7x faster** across all configs. With corrected FLOPS counting, FlashInfer peaks at **552 TFLOPS (55.8% of peak FP16)**, while Triton maxes at **209 TFLOPS (21.1% of peak)**. FlashInfer achieves over 2.5x the compute efficiency of Triton.

### 2.2 NCU Prefill Analysis (bs=4, seq=2048)

| Metric | FlashInfer | Triton |
|--------|-----------|--------|
| **Kernel** | `PrefillWithKVCacheKernel` (Hopper `CollectiveMainloop`) | `_fwd_kernel` |
| **DRAM throughput** | 13.1% of peak | 9.3% of peak |
| **SM throughput** | **52.0%** of peak | **32.7%** of peak |
| **Active warps** | 14.1% | 12.5% |
| **Registers/thread** | 168 | **255** (hardware max!) |
| **Block size** | 384 | 256 |
| **Grid size** | 132 (= #SMs, cooperative) | (4, 32, 16) = 2048 |
| **L1 global load sectors** | 108K total (97% hit) | 37.8M total (0.1% hit) |
| **L2 hit rate** | **86.4%** (34.4M hit / 5.4M miss) | **72.0%** (28.8M hit / 9.9M miss) |
| **Tensor core insts** | 2,228,224 | 2,228,224 |
| **Local memory (spills)** | n/a | n/a |
| **Duration** | 368 us | 909 us |

### 2.3 Why FlashInfer Wins Prefill So Decisively

1. **TMA vs global loads**: FlashInfer's Hopper kernel uses TMA (Tensor Memory Accelerator) for bulk data movement, which **bypasses the L1 global load path entirely**. This is why FlashInfer shows only 108K L1 global load sectors (residual metadata) vs Triton's 37.8M sectors. TMA loads go directly to shared memory through a dedicated hardware path, avoiding L1 cache pollution and achieving higher throughput. This is not "97% L1 hit rate" in the traditional sense — it's a fundamentally different data path.

2. **Register pressure**: Triton uses **255 registers/thread** (the sm_90 hardware maximum), while FlashInfer uses 168. Hitting the register ceiling strongly suggests the compiler couldn't reduce live variables further. However, NCU spill counters returned `n/a` for both kernels on Hopper, so we cannot directly confirm spilling from the available metrics. The register difference is real; its exact performance impact (occupancy-only vs occupancy + spill latency) requires further investigation.

3. **Cooperative grid scheduling**: FlashInfer launches exactly 132 blocks (one per SM) using CUDA cooperative launch semantics. This ensures every SM gets exactly one block, maximizing shared memory utilization and enabling persistent-kernel-style scheduling. Triton launches 2048 blocks — 15.5x more — creating wave quantization effects and potential L2 thrashing as different waves evict each other's cached data.

4. **L2 cache efficiency**: FlashInfer achieves **86.4% L2 hit rate** vs Triton's **72.0%**. With 132 cooperative blocks, FlashInfer's L2 working set is smaller and more controlled. Triton's 2048-block grid generates more L2 pressure, leading to 83% more L2 misses (9.9M vs 5.4M sectors).

5. **SM utilization**: FlashInfer hits **52% SM throughput** — solidly compute-bound. Triton at 32.7% is significantly underutilizing compute, likely due to the combination of register pressure, L2 misses, and wave scheduling overhead.

---

## 3. MLA (Multi-head Latent Attention) — DeepSeek-V2-Lite

| Batch Size | FlashInfer (ms) | Script BW (GB/s) | Actual BW (GB/s) |
|-----------|----------------|-------------------|------------------|
| 1 | 0.019 | 1,290 | ~183 |
| 4 | 0.019 | 5,245 | ~745 |
| 16 | 0.021 | 18,958 | ~2,694 |
| 64 | 0.065 | 24,755 | ~3,518 |
| 128 | 0.112 | 28,666 | ~4,074 |
| 256 | 0.214 | 30,160 | ~2,822 |

**Important caveat on bandwidth numbers**: The "Script BW" column uses the logical (expanded-head) KV size, which inflates bandwidth beyond physical HBM peak. The "Actual BW" column estimates real memory traffic using the compressed KV size: `bs * kv_len * (kv_lora_rank + qk_rope_head_dim) * 2 bytes = bs * kv_len * 576 * 2`.

For DeepSeek-V2-Lite's MLA config (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `num_kv_heads=16`, `head_dim=128`):
- **GQA equivalent KV**: `2 * bs * kv_len * 16 * 128 * 2 = 4.295 GB` at bs=256
- **Actual MLA compressed KV**: `bs * kv_len * (512 + 64) * 2 = 0.604 GB` at bs=256
- **Compression ratio**: GQA reads **7.1x more data** than MLA for the same logical attention

This is MLA's core advantage: by projecting K and V into a shared low-rank latent space, it dramatically reduces HBM traffic. The actual achieved bandwidth (~2,822 GB/s at bs=256) is 84% of HBM peak — confirming the kernel is well-optimized and memory-bound at scale.

---

## 4. End-to-End: Attention in Context (Qwen2.5-7B, decode, bs=64)

| Component | Time (us) | % of Total |
|-----------|----------|-----------|
| Linear layers (nvjet gemms) | 5,079 | 70.6% |
| **Attention (FlashAttn Hopper)** | **1,413** | **19.6%** |
| Norms (RMSNorm) | 145 | 2.0% |
| Activation (SiLU) | 98 | 1.4% |
| RoPE | 57 | 0.8% |
| KV cache store | 36 | 0.5% |
| Other | 363 | 5.0% |

**Attention accounts for ~20% of decode time** in the full model. Linear layers (GEMMs) dominate at 71%. This means a 10% improvement in attention kernel bandwidth translates to only ~2% end-to-end speedup — important context for the FlashInfer vs Triton comparison.

**Note**: This uses Qwen2.5-7B (non-gated) with dummy weights via `bench_one_batch`. Llama-3.1-8B requires HuggingFace authentication. The architectural similarity (GQA, similar dims) makes this a valid proxy.

---

## 5. Key Findings

### Finding 1: Decode attention is definitively memory-bound
Both FlashInfer (84% DRAM) and Triton (76% DRAM) are firmly memory-bandwidth limited during decode. SM utilization is 17% and 28% respectively. **Optimization should focus on reducing memory traffic**, not compute efficiency.

### Finding 2: FlashInfer's register-heavy strategy pays off for decode
FlashInfer trades 2.4x more registers (183 vs 76) for 10% higher DRAM throughput (84% vs 76%). Despite only 12% active warps (vs Triton's 35%), FlashInfer's single-kernel design with superior memory access patterns wins. The textbook says "maximize occupancy" — FlashInfer says "maximize bandwidth per warp."

### Finding 3: Prefill gap is larger and architecturally deeper than decode
FlashInfer's 2-2.7x prefill advantage stems from Hopper-specific hardware features unavailable to Triton: TMA loads (bypassing L1 global load path), cooperative grid launch (132 blocks = 1 per SM), and a Cutlass-based `CollectiveMainloop` that orchestrates data movement through shared memory. Triton's generic compiler cannot access TMA or cooperative launch, and hits the 255-register ceiling. This gap is architectural, not tuning.

### Finding 4: L2 cache doesn't help GQA decode at typical batch sizes
Despite Llama-3-8B's 4:1 GQA ratio, L2 hit rates are <1% for both backends during decode at bs=64. The KV cache (`64 * 2048 * 8 * 128 * 2 = 256 MB`) is 5x larger than L2 (50 MB). GQA's head sharing would only help if the KV cache fit in L2 — possible at bs=1 where KV is only 4 MB.

### Finding 5: MLA reduces memory traffic 7x vs GQA
MLA's compressed KV representation reads **7.1x less data** than GQA for the same logical attention (for DeepSeek-V2-Lite's config). The kernel achieves ~84% of HBM peak bandwidth on the compressed data, confirming efficient implementation.

---

## 6. Verification Notes

### What's solid
- **Decode timing and bandwidth**: Verified across 3 independent trials, <1% coefficient of variation at bs>=64. Bandwidth formula is standard (KV + Q + O bytes / time).
- **NCU kernel identification**: Correct kernels profiled — `BatchPrefillWithPagedKVCacheKernel` for FlashInfer (note: FlashInfer uses its prefill kernel for decode with page_size=1), `_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2` for Triton.
- **Register counts**: Directly from NCU `launch__registers_per_thread`. 183 vs 76 (decode), 168 vs 255 (prefill).
- **DRAM throughput %**: Directly from NCU `dram__throughput.avg.pct_of_peak_sustained_elapsed`.

### What needs caveats
- **L1 hit rate for prefill**: The "97% vs 0.1%" comparison is misleading as stated. FlashInfer's TMA loads bypass the L1 global load path — the 97% is on a tiny residual (108K sectors vs 37.8M). The correct framing: FlashInfer uses TMA (dedicated hardware path), Triton uses global loads (all L1 misses). The real cache comparison is L2: 86.4% vs 72%.
- **Register spilling**: NCU returned `n/a` for local memory wavefront counters on all Hopper kernels. We confirmed register counts differ (183 vs 76, 168 vs 255) but cannot directly measure spill traffic.
- **TFLOPS formula**: Original script undercounts by 2x (only QK^T, not AV). Corrected values in Section 2.1.
- **MLA bandwidth**: Script reports "effective BW" using logical tensor sizes (exceeds physical HBM peak). Corrected actual BW in Section 3.
- **GPU clocks**: Not locked during profiling. Adds ~1-3% variance to absolute numbers but does not affect relative comparisons (both backends run consecutively under same thermal conditions).

---

## 7. Optimization Opportunities

1. **Triton prefill register pressure**: The 255-register ceiling is the primary bottleneck. Restructuring the kernel to use fewer registers (e.g., different tiling, reducing live variables) could unlock significant gains. However, the deeper issue is lack of TMA access — Triton would need compiler-level support for Hopper TMA to match FlashInfer's prefill.

2. **Triton decode launch overhead**: At small batch sizes (bs=1-4), Triton's two-kernel approach adds disproportionate overhead. Fusing Stage 1 and Stage 2 would help latency-sensitive workloads.

3. **L2 cache exploitation for small-batch decode**: At bs=1 with kv_len=2048, the KV cache is only 4 MB — well within L2. A kernel that deliberately tiles to maximize L2 reuse across Q heads could improve small-batch decode latency significantly.

---

## Appendix A: Reproducibility Data

Three independent trials of key decode configs (warmup=20, iters=200):

| Config | Trial 1 | Trial 2 | Trial 3 | Spread |
|--------|---------|---------|---------|--------|
| FI decode bs=1 kv=2048 | 0.029ms | 0.030ms | 0.030ms | ±3.4% |
| Tri decode bs=1 kv=2048 | 0.056ms | 0.059ms | 0.057ms | ±2.6% |
| FI decode bs=64 kv=2048 | 0.187ms | 0.187ms | 0.187ms | <0.1% |
| Tri decode bs=64 kv=2048 | 0.216ms | 0.216ms | 0.216ms | <0.1% |
| FI decode bs=256 kv=2048 | 0.719ms | 0.718ms | 0.718ms | <0.1% |
| Tri decode bs=256 kv=2048 | 0.813ms | 0.813ms | 0.813ms | <0.1% |

At bs>=64, results are essentially deterministic (sub-0.1% variance). At bs=1, the ~3% spread is due to GPU boost clock variance on a short (~28us) kernel — both backends show similar variance, so relative speedup ratios are stable.

## Appendix B: Raw Data Files

```
results_decode_batchsweep.csv     # Decode batch size sweep
results_decode_kvsweep.csv        # Decode KV length sweep
results_mla_decode.csv            # MLA decode (FlashInfer only)
results_prefill.csv               # Prefill sweep
ncu_results/                      # NCU kernel-level reports
e2e_results/                      # Torch profiler traces (open in perfetto.dev)
```

## Appendix C: Reproduction Commands

```bash
cd ~/sglang/profiling

# Decode sweeps
python3 profile_attention_kernels.py --model llama-8b --mode decode --backend both \
    --batch-sizes 1,4,16,64,128,256 --kv-lens 2048 --output results_decode_batchsweep.csv

# KV sweep
python3 profile_attention_kernels.py --model llama-8b --mode decode --backend both \
    --batch-sizes 64 --kv-lens 256,512,1024,2048,4096,8192 --output results_decode_kvsweep.csv

# Prefill
python3 profile_attention_kernels.py --model llama-8b --mode prefill --backend both \
    --batch-sizes 1,4,16 --seq-lens 512,1024,2048,4096 --output results_prefill.csv

# NCU decode comparison
export PATH=/usr/local/cuda/bin:$PATH
echo -1 > /proc/sys/kernel/perf_event_paranoid

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
lts__t_sectors_op_read_lookup_hit.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
smsp__inst_executed_pipe_tensor.sum,\
launch__registers_per_thread,\
launch__block_size,launch__grid_size,\
gpu__time_duration.sum \
    --profile-from-start no --target-processes all --csv \
    python3 profile_attention_kernels.py --model llama-8b --mode decode \
    --backend flashinfer --batch-sizes 64 --kv-lens 2048 --ncu-mode \
    > ncu_results/flashinfer_decode.csv

# Same command with --backend triton for comparison
```
