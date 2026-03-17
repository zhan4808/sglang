"""
INT4 Perplexity Evaluation for MLA Reconstruction Weights
=========================================================
Evaluates whether INT4 quantization of kv_b_proj (reconstruction) weights
preserves model quality, as measured by wikitext-2 perplexity.

Three configs:
  1. FP16 baseline
  2. INT4 selective: only kv_b_proj weights quantized
  3. INT4 all linear weights

Target: <0.5 PPL delta for selective quantization.
"""

import torch
import gc
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
STRIDE = 512
MAX_LENGTH = 2048

def compute_perplexity(model, tokenizer, dataset_text, stride=STRIDE, max_length=MAX_LENGTH):
    """Sliding-window perplexity on wikitext-2 test set (standard HF approach)."""
    encodings = tokenizer("\n\n".join(dataset_text), return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    print(f"  Total tokens: {seq_len}")

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_chunk = input_ids[:, begin:end].to(model.device)

        # Build target: -100 for context tokens (overlap), real ids for new tokens
        target = input_chunk.clone()
        # Mask out the overlapping context (tokens we've already scored)
        if begin > 0:
            overlap = end - begin - stride  # how many tokens are reused context
            if overlap > 0:
                target[:, :overlap] = -100

        with torch.no_grad():
            outputs = model(input_chunk, use_cache=False, labels=target)
            # outputs.loss is mean over non-ignored tokens
            neg_log_likelihood = outputs.loss

        # Count non-masked tokens
        n_scored = (target != -100).sum().item() - 1  # -1 because labels are shifted internally
        nlls.append(neg_log_likelihood.item())

        prev_end = end
        if end >= seq_len:
            break

    # Average of per-window mean NLLs (approximate but standard)
    ppl = torch.exp(torch.tensor(sum(nlls) / len(nlls))).item()
    return ppl, seq_len


def quantize_selective_int4(model):
    """Quantize only kv_b_proj weights to INT4 (simulated via round-trip)."""
    count = 0
    for name, param in model.named_parameters():
        if "kv_b_proj" in name and param.ndim == 2:
            with torch.no_grad():
                # Per-channel asymmetric INT4 quantization (simulate)
                w = param.data
                ch_min = w.min(dim=1, keepdim=True).values
                ch_max = w.max(dim=1, keepdim=True).values
                scale = (ch_max - ch_min) / 15.0  # 4-bit = 16 levels
                scale = scale.clamp(min=1e-8)
                zero_point = ch_min
                w_q = ((w - zero_point) / scale).round().clamp(0, 15)
                w_deq = w_q * scale + zero_point
                param.data.copy_(w_deq)
            count += 1
    print(f"  Quantized {count} kv_b_proj weights to INT4 (simulated)")
    return model


def quantize_all_int4(model):
    """Quantize ALL linear weights to INT4 (simulated via round-trip)."""
    count = 0
    for name, param in model.named_parameters():
        if param.ndim == 2 and param.numel() >= 1024:  # skip tiny params
            with torch.no_grad():
                w = param.data
                ch_min = w.min(dim=1, keepdim=True).values if w.shape[0] <= w.shape[1] else w.min(dim=0, keepdim=True).values
                ch_max = w.max(dim=1, keepdim=True).values if w.shape[0] <= w.shape[1] else w.max(dim=0, keepdim=True).values
                scale = (ch_max - ch_min) / 15.0
                scale = scale.clamp(min=1e-8)
                zero_point = ch_min
                w_q = ((w - zero_point) / scale).round().clamp(0, 15)
                w_deq = w_q * scale + zero_point
                param.data.copy_(w_deq)
            count += 1
    print(f"  Quantized {count} linear weights to INT4 (simulated)")
    return model


def load_fresh_model():
    """Load a fresh FP16 model."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model


def run_eval():
    print("=" * 60)
    print("INT4 Perplexity Evaluation — DeepSeek-V2-Lite")
    print("=" * 60)

    # Load tokenizer and dataset
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading wikitext-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if t.strip()]
    print(f"  {len(texts)} non-empty lines")

    results = {}

    # ── Config 1: FP16 baseline ──
    print("\n" + "─" * 40)
    print("Config 1: FP16 Baseline")
    print("─" * 40)
    model = load_fresh_model()
    t0 = time.time()
    ppl, ntok = compute_perplexity(model, tokenizer, texts)
    elapsed = time.time() - t0
    print(f"  PPL = {ppl:.4f}  ({ntok} tokens, {elapsed:.1f}s)")
    results["fp16_baseline"] = {"ppl": ppl, "tokens": ntok, "time_s": elapsed}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Config 2: INT4 selective (kv_b_proj only) ──
    print("\n" + "─" * 40)
    print("Config 2: INT4 Selective (kv_b_proj only)")
    print("─" * 40)
    model = load_fresh_model()
    model = quantize_selective_int4(model)
    t0 = time.time()
    ppl_sel, ntok = compute_perplexity(model, tokenizer, texts)
    elapsed = time.time() - t0
    print(f"  PPL = {ppl_sel:.4f}  ({ntok} tokens, {elapsed:.1f}s)")
    results["int4_selective"] = {"ppl": ppl_sel, "tokens": ntok, "time_s": elapsed}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Config 3: INT4 all linear weights ──
    print("\n" + "─" * 40)
    print("Config 3: INT4 All Linear Weights")
    print("─" * 40)
    model = load_fresh_model()
    model = quantize_all_int4(model)
    t0 = time.time()
    ppl_all, ntok = compute_perplexity(model, tokenizer, texts)
    elapsed = time.time() - t0
    print(f"  PPL = {ppl_all:.4f}  ({ntok} tokens, {elapsed:.1f}s)")
    results["int4_all"] = {"ppl": ppl_all, "tokens": ntok, "time_s": elapsed}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    baseline = results["fp16_baseline"]["ppl"]
    sel = results["int4_selective"]["ppl"]
    all_ = results["int4_all"]["ppl"]
    print(f"  FP16 baseline:        {baseline:.4f}")
    print(f"  INT4 selective:       {sel:.4f}  (Δ = {sel - baseline:+.4f})")
    print(f"  INT4 all weights:     {all_:.4f}  (Δ = {all_ - baseline:+.4f})")
    print()
    if abs(sel - baseline) < 0.5:
        print("  ✓ Selective INT4 passes quality gate (<0.5 PPL delta)")
    else:
        print("  ✗ Selective INT4 FAILS quality gate (≥0.5 PPL delta)")
    print()

    # Save results
    out_path = "/root/sglang/profiling/results_int4_perplexity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_eval()
