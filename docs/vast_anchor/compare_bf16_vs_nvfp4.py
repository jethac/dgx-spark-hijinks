#!/usr/bin/env python3
"""Cross-compare bf16-cache vs nvfp4-cache serving, per captured layer, for 26B vs 12B.

The FlashInfer nvfp4 READER was proven faithful (compare_fi_vs_ref.py: out == dequant+SDPA).
So the 26B break must be in the cache CONTENTS / quantization sensitivity, not the kernel.
This isolates that: layer-0 q is KV-independent (pure embeddings->q_proj->qknorm->rope), so it is
IDENTICAL across a bf16 run and an nvfp4 run. Comparing layer-0 attention OUTPUT (bf16-cache vs
nvfp4-cache) measures the PURE per-layer quantization perturbation. If 26B >> 12B there, 26B is
genuinely more KV-quant-sensitive at this global scale (calibration / content story). For layers>0
the q drifts (depends on prior quantized attention) so the gap also includes accumulated drift —
reported as the trajectory-divergence onset.

Needs: cap_<m> (nvfp4 run, has q=arg0 + out) and cap_<m>_bf16 (bf16 run, q+out only).
"""
import glob
import os

import torch


def rel(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    mx = (a - b).abs().max().item()
    mn = (a - b).abs().mean().item()
    refm = b.abs().mean().item()
    return cos, mx, mn, (mn / refm if refm else float("nan"))


def load_qout(capdir):
    out = {}
    for f in sorted(glob.glob(os.path.join(capdir, "call_*.pt"))):
        d = torch.load(f, map_location="cpu")
        ci = d["_call_index"]
        out[ci] = (d["arg0"], d["out"])
    return out


def analyze(model_tag, label):
    nv = load_qout(f"/root/cap_{model_tag}")
    bf = load_qout(f"/root/cap_{model_tag}_bf16")
    common = sorted(set(nv) & set(bf))
    print(f"\n===== {label}: bf16-cache vs nvfp4-cache  (layers {common}) =====")
    print(f"{'layer':>5} {'q_shape':>16} | {'q nv-vs-bf cos':>14} {'q maxabs':>9} "
          f"| {'OUT nv-vs-bf cos':>16} {'out relerr':>10} {'out maxabs':>10}")
    for ci in common:
        qnv, onv = nv[ci]
        qbf, obf = bf[ci]
        qcos, qmx, _, _ = rel(qnv, qbf)
        ocos, omx, _, orel = rel(onv, obf)
        tag = "  <- layer0: KV-indep q (pure quant)" if ci == 0 else ""
        print(f"{ci:>5} {str(tuple(qnv.shape)):>16} | {qcos:>14.6f} {qmx:>9.4f} "
              f"| {ocos:>16.6f} {orel:>10.4%} {omx:>10.4f}{tag}")


if __name__ == "__main__":
    for tag, label in [("26b", "26B-A4B (broken)"), ("12b", "12B (control)")]:
        if os.path.isdir(f"/root/cap_{tag}_bf16"):
            try:
                analyze(tag, label)
            except Exception as e:
                import traceback
                print(f"{label}: ERROR {e}")
                traceback.print_exc()
