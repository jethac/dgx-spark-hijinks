#!/usr/bin/env python3
"""Resolve the refsim discrepancy: vast/Torch2.12 gave +0.1932, Spark/Torch2.11 gave +0.6949
for the SAME exact-reference nvfp4 q/dq. bf16 matched, only the q/dq side moved. Suspect: the
fp8-e4m3 block-scale conversion (`.to(torch.float8_e4m3fn)`) differs/bugs across Torch versions.

This runs the suffix-matched 12B-it reference with the block scale kept in THREE ways:
  SCALE=fp8  -> spec nvfp4 (torch.float8_e4m3fn)   [reproduces Codex's Spark number]
  SCALE=fp32 -> block scale in fp32 (no float8)    [isolates the float8 conversion]
  SCALE=manual -> deterministic e4m3 rounding via a representable-value table (version-indep)
Plus a tensor-level self-check: max abs diff between fp8-qdq and manual-qdq on a fixed tensor.
If fp32/manual land near +0.19 while fp8 is +0.69, the Torch float8 path is the culprit and the
true cost is ~+0.19. Env: REFMID, REFCTX=8185, REFPSTART=4096, SCALE in {fp8,fp32,manual}.
"""
import torch, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

tk = os.environ["HF_TOKEN"]
MID = os.environ.get("REFMID", "google/gemma-4-12B-it")
CTX = int(os.environ.get("REFCTX", "8185"))
PSTART = int(os.environ.get("REFPSTART", "4096"))
SCALE = os.environ.get("SCALE", "fp8")

E2M1 = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])


def _e4m3_table():
    # representable positive finite magnitudes of float8_e4m3fn (1-4-3, bias 7, max 448)
    vals = set([0.0])
    for e in range(0, 16):          # stored exponent 0..15
        for m in range(0, 8):       # 3 mantissa bits
            if e == 0:
                v = (m / 8.0) * 2.0 ** (1 - 7)          # subnormal
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)     # normal
            if v <= 448.0:
                vals.add(v)
    return torch.tensor(sorted(vals))


_E4M3 = _e4m3_table()


def to_e4m3_manual(x):  # round-to-nearest over the representable table (sign-symmetric)
    s = x.sign()
    a = x.abs().clamp(max=448.0)
    tbl = _E4M3.to(x.device)
    mids = (tbl[1:] + tbl[:-1]) / 2
    idx = torch.bucketize(a, mids)
    return s * tbl[idx]


def block_scale(amax):
    raw = (amax / 6.0).clamp(min=1e-8)
    if SCALE == "fp8":
        return raw.to(torch.float8_e4m3fn).float()
    if SCALE == "fp32":
        return raw
    if SCALE == "manual":
        return to_e4m3_manual(raw)
    raise SystemExit("bad SCALE")


def qdq_nvfp4(x):
    if x.shape[-1] % 16:
        return x
    lead = x.shape[:-1]; D = x.shape[-1]
    xb = x.reshape(*lead, D // 16, 16).float()
    amax = xb.abs().amax(-1, keepdim=True)
    scale = block_scale(amax)
    xs = xb / scale; sign = xs.sign(); mag = xs.abs().clamp(max=6.0)
    lv = E2M1.to(x.device); mids = (lv[1:] + lv[:-1]) / 2
    q = lv[torch.bucketize(mag, mids)] * sign
    return (q * scale).reshape(*lead, D).to(x.dtype)


# tensor-level self-check (does fp8 vs manual diverge on this box?)
torch.manual_seed(0)
_t = (torch.randn(4096, 16, device="cuda") * 0.1)
_raw = (_t.abs().amax(-1, keepdim=True) / 6.0).clamp(min=1e-8)
_fp8 = _raw.to(torch.float8_e4m3fn).float()
_man = to_e4m3_manual(_raw)
print(f"[selfcheck] torch.float8_e4m3fn vs manual-e4m3 on block scales: "
      f"maxabs={ (_fp8-_man).abs().max().item():.3e} mean={ (_fp8-_man).abs().mean().item():.3e}")

_orig = torch.nn.functional.scaled_dot_product_attention
MODE = [0]
def patched(q, k, v, *a, **kw):
    if MODE[0] == 1:
        k = qdq_nvfp4(k); v = qdq_nvfp4(v)
    return _orig(q, k, v, *a, **kw)
torch.nn.functional.scaled_dot_product_attention = patched

tok = AutoTokenizer.from_pretrained(MID, token=tk)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda", token=tk,
                                         attn_implementation="sdpa")
fp = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                     repo_type="dataset", token=tk)
text = "\n".join(x for x in pq.read_table(fp).column("text").to_pylist() if x and x.strip())
ids = tok(text)["input_ids"][:CTX]
inp = torch.tensor([ids], device="cuda")

def per_pos(mode):
    MODE[0] = mode
    with torch.no_grad():
        lg = m(inp).logits[0].float()
    lp = torch.log_softmax(lg[:-1], -1)
    return -lp[range(len(ids) - 1), torch.tensor(ids[1:], device="cuda")]

nb = per_pos(0); nq = per_pos(1)
s = slice(PSTART, len(ids) - 1)
b, q = nb[s].mean().item(), nq[s].mean().item()
print(f"SCALE={SCALE} model={MID} ctx={CTX} suffix=[{PSTART+1}..{CTX-1}]  bf16={b:.4f} nvfp4={q:.4f} DELTA={q-b:+.4f}")
