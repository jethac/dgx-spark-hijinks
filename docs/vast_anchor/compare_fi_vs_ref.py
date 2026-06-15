#!/usr/bin/env python3
"""Compare the REAL serving FlashInfer nvfp4 paged-attention output against a faithful
dequant+SDPA reference, per captured layer, for 26B (broken) vs 12B (control).

Inputs: the call_*.pt files written by docs/vast_anchor/sitecustomize.py, each holding the
exact wrapper.run() args for one scoring-prefill layer:
  arg0            = q          [S, Hq, Dqk]            bf16   (post-rope, post-qknorm query)
  arg1_0/arg1_1   = k_data/v_data  paged nvfp4 split views (uint8, packed 2 fp4/byte)
  kw_kv_cache_sf_0/_1 = k_sf/v_sf   per-16-block fp8 e4m3 scale views
  kw_k_scale/kw_v_scale = global scale (0.1 FIXSCALE)
  out             = FlashInfer attention output [S, Hq, Dvo]
  self_*          = plan state (page table indptr/indices/last_page_len, sm_scale, window_left)

Dequant + reference math are lifted verbatim from scripts/nvfp4_writer_roundtrip_probe.py
(validated: cosine>=0.9999 reader gate on 12B). If the kernel READ math is correct, FlashInfer
out == reference (small bf16 noise). 26B diverging while 12B matches => reader math bug.
"""
import glob
import os
import sys

import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
SF_BLOCK = 16


def dequant_pages(data, sf, gscale):
    """data uint8 [P,T,H,D/2], sf fp8 [P,T,H,D/16] -> bf16-ish [P,T,H,D] (probe _dequant_written_pages)."""
    lo = data & 0xF
    hi = (data >> 4) & 0xF
    idx = torch.stack((lo, hi), dim=-1).reshape(*data.shape[:-1], data.shape[-1] * 2)
    lut = torch.tensor(E2M1, dtype=torch.float32)
    vals = lut[idx.long()]
    sfe = sf.float().repeat_interleave(SF_BLOCK, dim=-1)
    return vals * sfe * float(gscale)


def gather_seq(sel, last_len):
    """sel = already-dequantized selected pages [npage, T, H, D] -> [seq, H, D]."""
    if sel.shape[0] == 1:
        return sel[0][:last_len]
    full = sel[:-1].reshape(-1, sel.shape[2], sel.shape[3])
    last = sel[-1][:last_len]
    return torch.cat([full, last], dim=0)


def sdpa_ref(q, k, v, sm_scale, window_left):
    """probe _torch_prefill_reference, but with the captured sm_scale (Gemma query_pre_attn_scalar)."""
    qo, Hq, _ = q.shape
    kv, Hkv, _ = k.shape
    g = Hq // Hkv
    kf = k.float().repeat_interleave(g, dim=1)
    vf = v.float().repeat_interleave(g, dim=1)
    qf = q.float()
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * sm_scale
    qpos = torch.arange(qo)[:, None]
    kpos = torch.arange(kv)[None, :]
    q_abs = kv - qo + qpos
    mask = kpos <= q_abs
    if window_left is not None and window_left >= 0:
        mask &= kpos + window_left >= q_abs
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vf)


def find_self(d, *needles):
    """Find a self_<name> entry whose key contains all needles (tensor) or its _repr (scalar)."""
    for k, v in d.items():
        if not k.startswith("self_"):
            continue
        kl = k.lower()
        if all(n in kl for n in needles):
            if k.endswith("_repr"):
                s = v
                try:
                    return float(s) if ("." in s or "e" in s.lower()) else int(s)
                except Exception:
                    return s
            return v
    return None


def metrics(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return cos, (a - b).abs().max().item(), (a - b).abs().mean().item(), b.abs().mean().item()


def analyze(capdir, label):
    files = sorted(glob.glob(os.path.join(capdir, "call_*.pt")))
    print(f"\n===== {label}  ({len(files)} captured layers) =====")
    print(f"{'call':>4} {'q_shape':>16} {'Dqk':>4} {'Dvo':>4} {'win':>5} {'sm':>8} "
          f"{'cosine':>9} {'maxabs':>9} {'meanabs':>9} {'ref|.|':>8}  verdict")
    for f in files:
        d = torch.load(f, map_location="cpu")
        q = d["arg0"]
        kd, vd = d["arg1_0"], d["arg1_1"]
        ksf, vsf = d["kw_kv_cache_sf_0"], d["kw_kv_cache_sf_1"]
        ks = float(d.get("kw_k_scale_repr", "1.0"))
        vs = float(d.get("kw_v_scale_repr", "1.0"))
        out = d["out"]
        sm = find_self(d, "sm_scale")
        if sm is None:
            sm = find_self(d, "scale") if not isinstance(find_self(d, "scale"), torch.Tensor) else None
        win = find_self(d, "window")
        indptr = find_self(d, "indptr") if find_self(d, "indptr") is not None else find_self(d, "kv_indptr")
        indices = find_self(d, "kv_indices")
        if indices is None:
            indices = find_self(d, "indices")
        last_len = find_self(d, "last_page")
        # single sequence (ctx 512): take seq 0's page range
        ip = indptr.long() if torch.is_tensor(indptr) else None
        if ip is not None and indices is not None:
            s0, s1 = int(ip[0]), int(ip[1])
            seq_pages = indices.long()[s0:s1]
        else:
            seq_pages = None
        ll = int(last_len[0]) if torch.is_tensor(last_len) else SF_BLOCK
        smv = float(sm) if sm is not None else (q.shape[-1] ** -0.5)
        winv = int(win) if win is not None else -1

        # call layout: sliding layers store [P, page=16, kvh=8, Dqk/2] (clean gather);
        # global VO-split (Dqk=512) store reshaped [P, 32, 2, ...] -> our simple gather
        # is NOT valid there, so flag those rather than trust the number.
        is_global = (kd.shape[1] != 16)
        # index the 32 sequence pages BEFORE dequant (full 148k-page cache is ~19GB dequantized)
        sp = seq_pages.long()
        kdq = dequant_pages(kd[sp], ksf[sp], ks)
        vdq = dequant_pages(vd[sp], vsf[sp], vs)
        kseq = gather_seq(kdq, ll)
        vseq = gather_seq(vdq, ll)
        # align kv length to q (causal end-aligned)
        ref = sdpa_ref(q, kseq, vseq, smv, winv)
        cos, mx, mn, refm = metrics(out, ref)
        if is_global:
            verdict = "global(geom?)"
        else:
            verdict = "MATCH" if cos > 0.999 and mx < 0.05 else ("DIVERGE" if cos < 0.99 else "soft")
        print(f"{d['_call_index']:>4} {str(tuple(q.shape)):>16} {q.shape[-1]:>4} {out.shape[-1]:>4} "
              f"{winv:>5} {smv:>8.5f} {cos:>9.5f} {mx:>9.4f} {mn:>9.5f} {refm:>8.4f}  {verdict}")


if __name__ == "__main__":
    for capdir, label in [("/root/cap_26b", "26B-A4B (broken)"), ("/root/cap_12b", "12B (control)")]:
        if os.path.isdir(capdir):
            try:
                analyze(capdir, label)
            except Exception as e:
                import traceback
                print(f"{label}: ERROR {e}")
                traceback.print_exc()
