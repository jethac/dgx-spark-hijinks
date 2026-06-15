#!/usr/bin/env python3
"""Phase 0 analysis: test the per-channel-outlier thesis on the captured post-RoPE K/V.

Two questions, both falsifiable:
  (1) DO outliers exist? Per-channel amax of K (over tokens x heads), then the WITHIN-16-BLOCK spread
      ratio = max(channel amax)/median(channel amax) per nvfp4 block. Outliers => some block has a
      channel >> its block-mates. Compare 26B layers 0-4 vs 12B same layers. If 26B's spread ratio is
      NOT worse than 12B's, the thesis is dead.
  (2) DOES a Hadamard rotation FIX it? Apply normalized Hadamard along head_dim, re-measure the block
      spread, AND an nvfp4 round-trip rel-L2 (per-16-block amax/6 e2m1 sim) rotated vs unrotated. The
      rotation must flatten the spread AND cut round-trip error to justify the build.

GATE for the kernel effort: 26B L0-4 spread >> 12B, AND rotation flattens it + cuts round-trip rel-L2.
"""
import glob
import os

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # positive nvfp4 levels


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)  # normalized: H @ H.T = I


def nvfp4_roundtrip(x):
    """Per-last-dim-16-block amax/6 e2m1 quant-dequant. x: [..., D]. Returns dequantized same shape."""
    D = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], D // 16, 16).float()
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 6.0
    q = xb / scale
    sgn = q.sign()
    mag = q.abs()
    idx = (mag.unsqueeze(-1) - E2M1).abs().argmin(-1)  # nearest level
    deq = sgn * E2M1[idx] * scale
    return deq.reshape_as(x)


def rel_l2(a, b):
    return ((a - b).float().norm() / a.float().norm().clamp_min(1e-12)).item()


def block_spread(x):
    """x: [N, D] -> per-16-block (max channel-amax / median channel-amax), return the WORST block ratio."""
    D = x.shape[-1]
    ch_amax = x.abs().amax(0)  # [D] per-channel amax over tokens*heads
    blk = ch_amax.reshape(D // 16, 16)
    ratio = blk.amax(-1) / blk.median(-1).values.clamp_min(1e-12)
    return ratio.max().item(), ratio.mean().item(), ch_amax.max().item(), ch_amax.median().item()


def analyze_dir(capdir, label):
    files = sorted(glob.glob(os.path.join(capdir, "wkv_*.pt")))
    print(f"\n===== {label}  ({len(files)} layers) =====")
    print(f"{'L':>2} {'D':>4} | {'K worstblk':>10} {'K maxch/med':>11} | "
          f"{'rot worstblk':>12} | {'nvfp4 relL2':>11} {'ROT relL2':>10} {'improve':>8}")
    for f in files:
        d = torch.load(f, map_location="cpu")
        K = d["key"].float()  # [T, H, D]
        T, H, D = K.shape
        Kf = K.reshape(T * H, D)
        worst, mean, mx, med = block_spread(Kf)
        # rotation
        Hd = hadamard(D)
        Krot = Kf @ Hd.t()
        rworst, rmean, _, _ = block_spread(Krot)
        # nvfp4 round-trip rel-L2 (per row's 16-blocks), unrotated vs rotated
        rt = rel_l2(Kf, nvfp4_roundtrip(Kf))
        rrt = rel_l2(Kf, (nvfp4_roundtrip(Krot)) @ Hd)  # quantize rotated, then un-rotate back to compare
        improve = rt / rrt if rrt else float("inf")
        maxch_med = (mx / med) if med else 0.0
        print(f"{d['i']:>2} {D:>4} | {worst:>10.2f} {maxch_med:>11.2f} | "
              f"{rworst:>12.2f} | {rt:>11.5f} {rrt:>10.5f} {improve:>7.2f}x")


if __name__ == "__main__":
    for tag, label in [("26b", "26B-A4B (broken) layers 0-7"), ("12b", "12B (control) layers 0-7")]:
        cd = f"/root/p0_{tag}"
        if os.path.isdir(cd):
            analyze_dir(cd, label)
    print("\nGATE: build the rotation IFF 26B L0-4 'K worstblk' >> 12B AND 'ROT relL2' << 'nvfp4 relL2' (improve>1).")
