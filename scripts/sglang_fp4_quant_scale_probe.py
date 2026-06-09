#!/usr/bin/env python3
"""Probe FlashInfer NVFP4 KV quantization scale conventions on the active GPU."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from flashinfer import fp4_quantize, nvfp4_kv_dequantize, nvfp4_kv_quantize


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    diff = (cand - ref).float()
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item()
    return {
        "shape": list(reference.shape),
        "max_abs": float(diff.abs().max()),
        "rms": float(torch.sqrt((diff * diff).mean())),
        "cosine": cosine,
        "reference_amax": float(ref.abs().max()),
        "candidate_amax": float(cand.abs().max()),
    }


def dequantize(fp4: torch.Tensor, scales: torch.Tensor, global_scale: float) -> torch.Tensor:
    global_scale_tensor = torch.tensor(
        [global_scale], dtype=torch.float32, device=fp4.device
    )
    return nvfp4_kv_dequantize(
        fp4.view(torch.uint8),
        scales.view(torch.uint8).reshape(fp4.shape[0], -1),
        global_scale_tensor,
        output_dtype=torch.bfloat16,
    )


def quantize_case(
    *,
    name: str,
    tensor: torch.Tensor,
    quant_global_scale: float,
    dequant_global_scale: float,
    is_global_scale_inversed: bool = False,
) -> dict[str, Any]:
    fp4, scales = fp4_quantize(
        tensor,
        torch.tensor([quant_global_scale], dtype=torch.float32, device=tensor.device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        is_sf_8x4_layout=False,
        is_global_scale_inversed=is_global_scale_inversed,
        enable_pdl=None,
    )
    dequant = dequantize(fp4, scales, dequant_global_scale)
    return {
        "name": name,
        "quant_global_scale": quant_global_scale,
        "dequant_global_scale": dequant_global_scale,
        "is_global_scale_inversed": is_global_scale_inversed,
        "comparison": compare_tensors(tensor, dequant),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tensor = torch.randn(
        args.rows, args.head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.5
    tensor[0, 0] = 4.375
    if args.rows > 1 and args.head_dim > 3:
        tensor[1, 3] = -3.75

    amax = float(tensor.float().abs().amax())
    current_dequant_scale = amax / (6.0 * 448.0)
    helper_dequant_scale = amax / 448.0

    cases = [
        quantize_case(
            name="sglang_current_inverse_arg_no_inverse_flag",
            tensor=tensor,
            quant_global_scale=1.0 / current_dequant_scale,
            dequant_global_scale=current_dequant_scale,
        ),
        quantize_case(
            name="sglang_current_inverse_arg_with_inverse_flag",
            tensor=tensor,
            quant_global_scale=1.0 / current_dequant_scale,
            dequant_global_scale=current_dequant_scale,
            is_global_scale_inversed=True,
        ),
        quantize_case(
            name="flashinfer_helper_quant_448_over_amax_dequant_amax_over_448",
            tensor=tensor,
            quant_global_scale=448.0 / amax,
            dequant_global_scale=helper_dequant_scale,
        ),
        quantize_case(
            name="flashinfer_helper_with_inverse_flag",
            tensor=tensor,
            quant_global_scale=448.0 / amax,
            dequant_global_scale=helper_dequant_scale,
            is_global_scale_inversed=True,
        ),
    ]

    fp4, scales = nvfp4_kv_quantize(
        tensor,
        torch.tensor([helper_dequant_scale], dtype=torch.float32, device=tensor.device),
    )
    cases.append(
        {
            "name": "nvfp4_kv_quantize_direct_dequant_amax_over_448",
            "quant_global_scale": helper_dequant_scale,
            "dequant_global_scale": helper_dequant_scale,
            "comparison": compare_tensors(tensor, dequantize(fp4, scales, helper_dequant_scale)),
        }
    )

    result = {
        "schema": "sglang-fp4-quant-scale-probe/v1",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
        "rows": args.rows,
        "head_dim": args.head_dim,
        "seed": args.seed,
        "amax": amax,
        "cases": cases,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
