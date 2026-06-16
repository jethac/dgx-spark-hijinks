#!/usr/bin/env python3
"""Compare bf16/NVFP4 active-KV captures with mixed K/V references.

The input directories are produced by ``active_kv_capture_sitecustomize.py``.
For matching FlashInfer prefill calls, this script reconstructs attention using:

* bf16 K + bf16 V
* NVFP4 K + NVFP4 V
* bf16 K + NVFP4 V
* NVFP4 K + bf16 V

This isolates whether the local attention perturbation is mainly QK/logit
driven (K) or value-path driven (V) without needing a full mixed-dtype vLLM
serving path.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any

import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
SF_BLOCK = 16


def dequant_pages(data: torch.Tensor, sf: torch.Tensor, gscale: float) -> torch.Tensor:
    lo = data & 0xF
    hi = (data >> 4) & 0xF
    idx = torch.stack((lo, hi), dim=-1).reshape(*data.shape[:-1], data.shape[-1] * 2)
    lut = torch.tensor(E2M1, dtype=torch.float32)
    vals = lut[idx.long()]
    return vals * sf.float().repeat_interleave(SF_BLOCK, dim=-1) * float(gscale)


def gather_seq(active_pages: torch.Tensor, last_len: int | None) -> torch.Tensor:
    if active_pages.dim() != 4:
        raise ValueError(f"expected active pages [P,T,H,D], got {tuple(active_pages.shape)}")
    if last_len is None:
        last_len = int(active_pages.shape[1])
    if active_pages.shape[0] == 1:
        return active_pages[0, :last_len]
    full = active_pages[:-1].reshape(-1, active_pages.shape[2], active_pages.shape[3])
    last = active_pages[-1, :last_len]
    return torch.cat([full, last], dim=0)


def sdpa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, window_left: int) -> torch.Tensor:
    qo, hq, _ = q.shape
    kv, hkv, _ = k.shape
    if hq % hkv:
        raise ValueError(f"Hq={hq} is not divisible by Hkv={hkv}")
    group = hq // hkv
    qf = q.float()
    kf = k.float().repeat_interleave(group, dim=1)
    vf = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * sm_scale
    qpos = torch.arange(qo)[:, None]
    kpos = torch.arange(kv)[None, :]
    q_abs = kv - qo + qpos
    mask = kpos <= q_abs
    if window_left >= 0:
        mask &= kpos + window_left >= q_abs
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vf)


def metrics(candidate: torch.Tensor, base: torch.Tensor) -> tuple[float, float, float, float]:
    a = candidate.detach().float().reshape(-1)
    b = base.detach().float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    diff = a - b
    rel = diff.norm().item() / max(b.norm().item(), 1e-12)
    return cos, rel, diff.abs().mean().item(), diff.abs().max().item()


def scalar_gain_metrics(candidate: torch.Tensor, base: torch.Tensor) -> tuple[float, float]:
    """Return best global gain alpha and rel-L2 of alpha * candidate vs base."""
    a = candidate.detach().float().reshape(-1)
    b = base.detach().float().reshape(-1)
    denom = torch.dot(a, a).item()
    if denom <= 1e-30:
        return float("nan"), float("inf")
    alpha = torch.dot(a, b).item() / denom
    rel = (a.mul(alpha) - b).norm().item() / max(b.norm().item(), 1e-12)
    return alpha, rel


def per_head_gain_metrics(candidate: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    """Fit one output gain per query head and report residual + gain stats."""
    if candidate.shape != base.shape or candidate.dim() != 3:
        return {
            "rel_l2": float("nan"),
            "alpha_mean": float("nan"),
            "alpha_std": float("nan"),
            "alpha_min": float("nan"),
            "alpha_max": float("nan"),
        }
    a = candidate.detach().float()
    b = base.detach().float()
    denom = torch.sum(a * a, dim=(0, 2)).clamp_min(1e-30)
    alpha = torch.sum(a * b, dim=(0, 2)) / denom
    scaled = a * alpha.view(1, -1, 1)
    rel = (scaled - b).reshape(-1).norm().item() / max(b.reshape(-1).norm().item(), 1e-12)
    return {
        "rel_l2": rel,
        "alpha_mean": alpha.mean().item(),
        "alpha_std": alpha.std(unbiased=False).item(),
        "alpha_min": alpha.min().item(),
        "alpha_max": alpha.max().item(),
    }


def first_tensor(d: dict[str, Any], names: tuple[str, ...]) -> torch.Tensor:
    for name in names:
        value = d.get(name)
        if torch.is_tensor(value):
            return value
    raise KeyError(f"none of {names} present")


def kv_from_bf16(d: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # Expected tuple-KV layout from FlashInfer: arg1_0_active = K, arg1_1_active = V.
    if torch.is_tensor(d.get("arg1_0_active")) and torch.is_tensor(d.get("arg1_1_active")):
        return d["arg1_0_active"], d["arg1_1_active"]
    if torch.is_tensor(d.get("kw_paged_kv_cache_0_active")) and torch.is_tensor(
        d.get("kw_paged_kv_cache_1_active")
    ):
        return d["kw_paged_kv_cache_0_active"], d["kw_paged_kv_cache_1_active"]
    # Fallback for packed [2, P, T, H, D] style.
    t = d.get("arg1_active")
    if torch.is_tensor(t) and t.dim() == 5 and t.shape[0] == 2:
        return t[0], t[1]
    # Fallback for packed [P, 2, T, H, D] style.
    if torch.is_tensor(t) and t.dim() == 5 and t.shape[1] == 2:
        return t[:, 0], t[:, 1]
    for key, value in d.items():
        if not key.endswith("_active") or not torch.is_tensor(value):
            continue
        if value.dim() != 5:
            continue
        if value.shape[0] == 2:
            return value[0], value[1]
        if value.shape[1] == 2:
            return value[:, 0], value[:, 1]
    candidates = [
        (key, value)
        for key, value in d.items()
        if key.endswith("_active")
        and torch.is_tensor(value)
        and value.dim() == 4
        and value.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ]
    candidates.sort(key=lambda kv: kv[0])
    if len(candidates) >= 2:
        return candidates[0][1], candidates[1][1]
    active = []
    for key, value in sorted(d.items()):
        if key.endswith("_active") and torch.is_tensor(value):
            active.append(f"{key}:{tuple(value.shape)}:{value.dtype}")
    raise KeyError("could not locate bf16 active K/V tensors; active=" + ",".join(active))


def kv_from_nvfp4(d: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    kd = first_tensor(d, ("arg1_0_active", "kw_paged_kv_cache_0_active"))
    vd = first_tensor(d, ("arg1_1_active", "kw_paged_kv_cache_1_active"))
    ksf = first_tensor(d, ("kw_kv_cache_sf_0_active", "kw_maybe_k_cache_sf_active", "kw_k_cache_sf_active"))
    vsf = first_tensor(d, ("kw_kv_cache_sf_1_active", "kw_maybe_v_cache_sf_active", "kw_v_cache_sf_active"))
    ks = float(d.get("kw_k_scale_repr", d.get("kw_k_scale", 1.0)))
    vs = float(d.get("kw_v_scale_repr", d.get("kw_v_scale", 1.0)))
    return dequant_pages(kd, ksf, ks), dequant_pages(vd, vsf, vs)


def load_files(path: Path) -> list[dict[str, Any]]:
    return [torch.load(p, map_location="cpu") for p in sorted(glob.glob(str(path / "call_*.pt")))]


def analyze_pair(bf: dict[str, Any], nv: dict[str, Any]) -> dict[str, Any]:
    q = bf["arg0"].float()
    bf_k_pages, bf_v_pages = kv_from_bf16(bf)
    nv_k_pages, nv_v_pages = kv_from_nvfp4(nv)
    last_len = int(bf.get("last_page_len") or nv.get("last_page_len") or bf_k_pages.shape[1])
    sm_scale = float(bf.get("sm_scale") or nv.get("sm_scale") or (q.shape[-1] ** -0.5))
    window_left = int(bf.get("window_left", nv.get("window_left", -1)))

    # This comparator intentionally handles the normal sliding page shape first.
    if bf_k_pages.dim() != 4 or nv_k_pages.dim() != 4:
        raise ValueError(f"unsupported page dims bf={bf_k_pages.shape} nv={nv_k_pages.shape}")
    if bf_k_pages.shape[1] != nv_k_pages.shape[1]:
        raise ValueError(f"page size mismatch bf={bf_k_pages.shape} nv={nv_k_pages.shape}")

    bf_k = gather_seq(bf_k_pages, last_len)
    bf_v = gather_seq(bf_v_pages, last_len)
    nv_k = gather_seq(nv_k_pages, last_len)
    nv_v = gather_seq(nv_v_pages, last_len)

    base = sdpa_ref(q, bf_k, bf_v, sm_scale, window_left)
    rows = {
        "nvfp4_kv": sdpa_ref(q, nv_k, nv_v, sm_scale, window_left),
        "bf16k_nvfp4v": sdpa_ref(q, bf_k, nv_v, sm_scale, window_left),
        "nvfp4k_bf16v": sdpa_ref(q, nv_k, bf_v, sm_scale, window_left),
    }
    out: dict[str, Any] = {
        "call": int(bf.get("_call_index", -1)),
        "q_shape": tuple(q.shape),
        "kv_tokens": int(bf_k.shape[0]),
        "window_left": window_left,
        "sm_scale": sm_scale,
    }
    for label, tensor in rows.items():
        cos, rel, mean_abs, max_abs = metrics(tensor, base)
        out[f"{label}_cos"] = cos
        out[f"{label}_rel_l2"] = rel
        out[f"{label}_mean_abs"] = mean_abs
        out[f"{label}_max_abs"] = max_abs
        alpha, gain_rel = scalar_gain_metrics(tensor, base)
        out[f"{label}_gain_alpha"] = alpha
        out[f"{label}_gain_rel_l2"] = gain_rel
        head = per_head_gain_metrics(tensor, base)
        out[f"{label}_head_gain_rel_l2"] = head["rel_l2"]
        out[f"{label}_head_gain_alpha_mean"] = head["alpha_mean"]
        out[f"{label}_head_gain_alpha_std"] = head["alpha_std"]
        out[f"{label}_head_gain_alpha_min"] = head["alpha_min"]
        out[f"{label}_head_gain_alpha_max"] = head["alpha_max"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True, type=Path)
    parser.add_argument("--nvfp4", required=True, type=Path)
    args = parser.parse_args()

    bf_files = load_files(args.bf16)
    nv_files = load_files(args.nvfp4)
    n = min(len(bf_files), len(nv_files))
    print(
        "call\tq_shape\tkv_tokens\twindow_left\tsm_scale\t"
        "nvfp4_kv_cos\tnvfp4_kv_rel_l2\t"
        "nvfp4_kv_gain_alpha\tnvfp4_kv_gain_rel_l2\t"
        "nvfp4_kv_head_gain_rel_l2\tnvfp4_kv_head_gain_alpha_mean\t"
        "nvfp4_kv_head_gain_alpha_std\t"
        "bf16k_nvfp4v_cos\tbf16k_nvfp4v_rel_l2\t"
        "nvfp4k_bf16v_cos\tnvfp4k_bf16v_rel_l2\tdominant"
    )
    for i in range(n):
        try:
            row = analyze_pair(bf_files[i], nv_files[i])
        except Exception as exc:
            print(f"{i}\tSKIP\t0\t0\tnan\tERROR\t{exc}")
            continue
        k_rel = row["nvfp4k_bf16v_rel_l2"]
        v_rel = row["bf16k_nvfp4v_rel_l2"]
        if math.isfinite(k_rel) and math.isfinite(v_rel):
            dominant = "K" if k_rel > v_rel * 1.25 else ("V" if v_rel > k_rel * 1.25 else "mixed")
        else:
            dominant = "unknown"
        print(
            f"{row['call']}\t{row['q_shape']}\t{row['kv_tokens']}\t"
            f"{row['window_left']}\t{row['sm_scale']:.8f}\t"
            f"{row['nvfp4_kv_cos']:.9f}\t{row['nvfp4_kv_rel_l2']:.9f}\t"
            f"{row['nvfp4_kv_gain_alpha']:.9f}\t{row['nvfp4_kv_gain_rel_l2']:.9f}\t"
            f"{row['nvfp4_kv_head_gain_rel_l2']:.9f}\t"
            f"{row['nvfp4_kv_head_gain_alpha_mean']:.9f}\t"
            f"{row['nvfp4_kv_head_gain_alpha_std']:.9f}\t"
            f"{row['bf16k_nvfp4v_cos']:.9f}\t{row['bf16k_nvfp4v_rel_l2']:.9f}\t"
            f"{row['nvfp4k_bf16v_cos']:.9f}\t{row['nvfp4k_bf16v_rel_l2']:.9f}\t"
            f"{dominant}"
        )


if __name__ == "__main__":
    main()
