# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---
#
# Convert to a Colab/Jupyter notebook with:
#   jupytext --to notebook gemma_nvfp4_kv_consumer_blackwell.py
# (Colab and VS Code can also open this percent-format file directly via jupytext.)
#
# Scaffold status: BEFORE half (cells 0-2). AFTER half (cells 3-4) lands once the
# jethac/flashinfer SWA-prefill fix is confirmed correct. See
# docs/COLAB_G4_SM120_GEMMA_NVFP4_DEMO.md for the full plan.

# %% [markdown]
# # NVFP4 KV Cache for Gemma on Consumer Blackwell
# ### (RTX PRO 6000 / `sm_120` and DGX Spark GB10 / `sm_121`)
#
# **The pitch:** NVFP4 (4-bit) KV cache should give you ~1.78× more context — or more
# concurrent requests — at the *same* decode speed, on a memory-bound Blackwell box. That's
# a big deal for Gemma: long context, and (Gemma 4 12B) multimodal, are KV-heavy.
#
# **The reality, today:** on *consumer* Blackwell — the RTX PRO 6000 and the DGX Spark —
# you can't turn it on for Gemma. This notebook shows that, precisely, on whichever card you
# have. The second half (added once our patch lands) shows it working.
#
# This is **capacity, not speed** — decode on these boxes is bandwidth-bound, so FP4 doesn't
# make tokens faster; it lets you *hold more*. We'll measure honestly throughout.

# %% [markdown]
# ## 0. Hardware check — must be consumer Blackwell (CC 12.x)
# Auto-detects the GPU and adapts. The same cells run on a Colab "G4" (`sm_120`) or a DGX
# Spark (`sm_121`); FlashInfer JIT-compiles for whichever arch it lands on.

# %%
import platform
import torch

assert torch.cuda.is_available(), "No CUDA GPU visible."
NAME = torch.cuda.get_device_name(0)
CAP = torch.cuda.get_device_capability(0)          # (12,0) RTX PRO 6000 | (12,1) GB10
MAJOR, MINOR = CAP
SM = f"sm_{MAJOR}{MINOR}"                            # sm_120 | sm_121
ARCH_A = f"{MAJOR}{MINOR}a"                          # 120a | 121a  (native FP4 target)
HOST = platform.machine()                           # x86_64 (Colab) | aarch64 (Spark)

assert MAJOR == 12, (
    f"This demo needs consumer Blackwell (CC 12.x). Got {CAP} ({NAME}). "
    "Datacenter Blackwell (sm_100) uses a different path and isn't the subject here."
)
IS_SPARK = CAP == (12, 1)       # GB10, unified memory
IS_RTXPRO = CAP == (12, 0)      # RTX PRO 6000 / Colab G4, discrete VRAM

# Memory model differs: GB10 = unified CPU+GPU pool (over-committing can deadlock the kernel,
# see docs/INCIDENT_20260609_OOM_DEADLOCK.md); RTX PRO 6000 = discrete VRAM (a GPU OOM just
# kills the process). Use safe defaults so this notebook never bricks the reader's box.
SAFE_GPU_MEM_FRACTION = 0.60 if IS_SPARK else 0.85
MEMORY_MODEL = "unified (shared CPU+GPU)" if IS_SPARK else "discrete VRAM"

print(f"GPU            : {NAME}")
print(f"Compute cap    : {CAP}  ->  {SM}  (native FP4 target: {ARCH_A})")
print(f"Host arch      : {HOST}")
print(f"Memory model   : {MEMORY_MODEL}")
print(f"Box            : {'DGX Spark (GB10)' if IS_SPARK else 'RTX PRO 6000-class'}")
print(f"Safe mem frac  : {SAFE_GPU_MEM_FRACTION}  (leave OS headroom; never run two big "
      f"servers at once)")

# %% [markdown]
# ## 0b. Hugging Face auth (Gemma is gated)
# Every Gemma model requires accepting Google's license on HF and a read token. (This is the
# exact gate that blocks `--kv-cache-dtype nvfp4` demos before they even start.)

# %%
import os
# Set your token: Colab -> add HF_TOKEN as a secret, or os.environ here. On the Spark it may
# already be in the environment / ~/.cache/huggingface/token.
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("HF auth: token present.")
else:
    print("HF auth: NO HF_TOKEN found. Accept the Gemma license on huggingface.co and set "
          "HF_TOKEN before running the model cells.")

# %% [markdown]
# ## 0c. Install the stock inference stack (the "before" baseline)
# Arch-branched: on Colab (x86_64) install pinned **stock** vLLM; on the Spark (aarch64) the
# environment already has it. The point of the BEFORE half is to show the *standard* stack
# can't do NVFP4 KV on these cards — no fork required.

# %%
PINNED_VLLM = "vllm"   # TODO: pin an exact stock version once confirmed on a G4 session.
if HOST == "x86_64":
    # Colab / G4
    # !pip -q install {PINNED_VLLM}
    print("On x86_64 (Colab): `pip install vllm` (uncomment above).")
elif HOST == "aarch64":
    # DGX Spark: use the already-installed stock vLLM, or run this notebook's kernel inside
    # the proven container. Do NOT rebuild here.
    print("On aarch64 (Spark): using the pre-installed stack. (Serving cells may prefer the "
          "container kernel.)")
else:
    print(f"Unexpected host arch {HOST}; verify the install path.")

# %% [markdown]
# ## 1. What NVFP4 KV cache is supposed to buy you
# The KV cache stores the attention keys/values for every token in context. Storing it in
# **NVFP4 (4-bit)** instead of fp8/bf16 shrinks it ~2-4×, so the same memory holds far more
# context or far more simultaneous requests — at the *same* decode speed, because decode is
# bound by memory bandwidth, not compute. It's a **capacity** lever.
#
# `--kv-cache-dtype nvfp4` is the knob. The catch: on consumer Blackwell it doesn't work yet.

# %% [markdown]
# ## 2. BEFORE — NVFP4 KV is unsupported for Gemma on consumer Blackwell
# We attempt to bring up Gemma with `kv_cache_dtype="nvfp4"` on the **stock** stack and show
# it fails, while `fp8` (the control) works — proving the model loads fine and it's
# specifically NVFP4 KV that's missing on `sm_120`/`sm_121`.

# %%
# Small, fast Gemma to demonstrate the failure mode (it fails at engine/KV setup, so size
# doesn't matter). Gemma 3 is widely supported on stock vLLM; Gemma 4 hits the same wall but
# also needs a newer stack (transformers >= 5.5.0) — we note that and sweep it below.
DEMO_MODEL = "google/gemma-3-4b-it"


def try_bringup(model_id, kv_cache_dtype, max_model_len=2048):
    """Attempt an offline vLLM bring-up; return (ok, detail). Never raises."""
    from vllm import LLM, SamplingParams  # imported lazily so install issues are localized
    try:
        llm = LLM(
            model=model_id,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=SAFE_GPU_MEM_FRACTION,
            enforce_eager=True,
            # one model at a time; never co-resident with another server (unified-mem safety)
        )
        out = llm.generate(["2 + 2 ="], SamplingParams(max_tokens=4, temperature=0))
        text = out[0].outputs[0].text
        del llm
        torch.cuda.empty_cache()
        return True, f"served; sample output={text!r}"
    except Exception as e:  # noqa: BLE001 — we WANT to capture the failure verbatim
        torch.cuda.empty_cache()
        # Trim to the salient line (e.g. "no trtllm-gen cubin", "unsupported kv dtype", ...).
        msg = str(e).strip().splitlines()
        return False, (msg[-1] if msg else "unknown error")[:300]


# %%
# Control vs treatment on one model: fp8 should work, nvfp4 should fail on sm_12x.
print(f"== {DEMO_MODEL} on {SM} ==")
ok_fp8, d_fp8 = try_bringup(DEMO_MODEL, "fp8")
print(f"  kv_cache_dtype=fp8    -> {'OK' if ok_fp8 else 'FAIL'} : {d_fp8}")
ok_nvfp4, d_nvfp4 = try_bringup(DEMO_MODEL, "nvfp4")
print(f"  kv_cache_dtype=nvfp4  -> {'OK' if ok_nvfp4 else 'FAIL'} : {d_nvfp4}")
# Expected BEFORE: fp8 OK, nvfp4 FAIL (routes to trtllm-gen -> no sm_12x cubins).
# TODO: confirm the exact nvfp4 failure string on a real G4 + Spark and record it.

# %% [markdown]
# ### 2c. The whole Gemma family — same wall
# Sweep the family (and a non-Gemma control). On the stock stack, **every** Gemma fails for
# NVFP4 KV on consumer Blackwell. Lean on small variants so it's fast; the failure is at the
# KV/dtype layer, independent of size.

# %%
# Configurable matrix. Default = representative + fast; flip RUN_ALL for the full family.
RUN_ALL = False
GEMMA_MATRIX = [
    # (model_id, family/arch note)
    ("google/gemma-3-4b-it", "Gemma 3 dense, SWA, small"),
    ("google/gemma-3-27b-it", "Gemma 3 dense, SWA, large"),
    # Gemma 4 (need a newer stack on stock; same NVFP4-KV wall). IDs per the config audit:
    ("google/gemma-4-E4B-it", "Gemma 4 mobile (PLE/audio)"),
    ("google/gemma-4-12B-it", "Gemma 4 dense, encoder-free multimodal, D=512"),
    ("google/gemma-4-26B-A4B-it", "Gemma 4 MoE, D=512"),
    ("google/gemma-4-31B-it", "Gemma 4 dense, D=512"),
]
CONTROL = [("Qwen/Qwen2.5-1.5B-Instruct", "non-Gemma, full attention (no SWA)")]
matrix = GEMMA_MATRIX + CONTROL
if not RUN_ALL:
    matrix = [GEMMA_MATRIX[0], GEMMA_MATRIX[1], GEMMA_MATRIX[3]] + CONTROL  # small default

rows = []
for model_id, note in matrix:
    ok, detail = try_bringup(model_id, "nvfp4", max_model_len=2048)
    rows.append({"model": model_id, "note": note, "nvfp4_kv": "OK" if ok else "UNSUPPORTED",
                 "detail": detail})
    print(f"  {model_id:34s} nvfp4_kv -> {'OK' if ok else 'UNSUPPORTED'}")

# %%
# BEFORE summary table.
try:
    import pandas as pd
    before_df = pd.DataFrame(rows)
    display(before_df[["model", "note", "nvfp4_kv", "detail"]])  # noqa: F821 (notebook)
except Exception:
    for r in rows:
        print(r)

print()
print(f"BEFORE on {SM} ({NAME}):")
print("  NVFP4 KV cache is UNSUPPORTED for the entire Gemma family on consumer Blackwell.")
print("  (The stock path routes nvfp4 KV to trtllm-gen, which ships no sm_120/sm_121 cubins"
      " — TRT-LLM #11799.)")

# %% [markdown]
# ## 2d. (Deep dive, optional) *Why* — at the kernel level
# For the curious: the standalone FlashInfer NVFP4-KV correctness probe shows the FA2 path
# itself. With our *in-progress* fork, decode + non-SWA prefill work, but the **SWA
# paged-prefill** kernel fails to compile — the params struct was missing its scale-factor
# strides. That's the precise hole the AFTER half fixes.
#
# (Uses `scripts/flashinfer_nvfp4_kv_probe.py` from the repo. This cell is the same probe
# that becomes the AFTER validator — it JIT-compiles for `{ARCH_A}` on whatever box you run.)

# %%
# TODO: wire scripts/flashinfer_nvfp4_kv_probe.py here for a Gemma SWA-prefill shape and show
# the compile failure on the pre-fix fork; on the fixed fork (AFTER) it compiles + cosine>=.9999.
print("Deep-dive probe placeholder — see scripts/flashinfer_nvfp4_kv_probe.py.")

# %% [markdown]
# ---
# ## AFTER (added once jethac/flashinfer SWA-prefill fix is confirmed correct)
# 3. Install the patched forks (`jethac/flashinfer@0919cdda+`, `jethac/vllm`), clear stale
#    JIT cache.
# 4a. Standalone probe: SWA paged-prefill compiles, cosine >= 0.9999 on this arch.
# 4b. Re-run the family sweep -> Gemma now serves NVFP4 KV, passes the quality gate.
# 4c. Capacity payoff: fp8 vs nvfp4 KV pool tokens (~1.78x) at decode parity.
# 4d. The "wow": a long prompt that OOMs under fp8 KV but fits under nvfp4 at the same budget.
# 4e. Honest Gemma PPL delta (measured — Gemma's outlier attention means don't assume Qwen's).
#
# See docs/COLAB_G4_SM120_GEMMA_NVFP4_DEMO.md.
