# FlashInfer Attention Kernel Analysis: Triton decode bs=64 kv=2048

## Kernel Summary

Total unique kernels: 1

### #1: `unknown`

- **Invocations**: 9
- **Classification**: **LATENCY-BOUND**
- SM throughput: 0.0% of peak
- Memory throughput: 0.0% of peak

> **Observation**: This kernel has low utilization of both compute and memory. Likely launch overhead, synchronization, or small problem size. Consider: kernel fusion, increasing batch size, or reducing launch overhead.

## Overall Assessment

- LATENCY-BOUND: 1 kernels

## Key Metrics to Report

When writing up the analysis, focus on:
1. **Decode attention**: Is it memory-bound? What % of peak DRAM bandwidth?
2. **Prefill attention**: Is it compute-bound? What % of peak TFLOPS?
3. **Occupancy**: Low occupancy often means register pressure or shared memory limits
4. **L2 cache**: KV cache access patterns - is L2 helping or thrashing?
5. **MLA vs GQA**: Does MLA's compressed KV actually reduce memory traffic?
