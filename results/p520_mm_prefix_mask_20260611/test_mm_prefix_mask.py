#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mm-prefix span-level bidirectional masking: FlashInfer packed-custom-mask
probe gate for the vLLM FlashInfer backend (spark-hijinks epoch-2).

Mimics the planned vLLM FlashInferMetadataBuilder mm-prefix grouping over a
synthetic mixed batch WITHOUT serving: requests whose image span intersects
the query window go to a second BatchPrefillWithPagedKVCacheWrapper planned
with `custom_mask` encoding (causal AND sliding-window) OR (q in span AND
kv in span) -- the mask-order contract from vLLM's FlexAttention /
triton_unified_attention mm-prefix paths (spans inclusive [start, end] in
document positions). Plain requests stay on the fast causal wrapper.
Outputs are gathered/scattered by token index exactly like the DG-2
causal-grouping path in FlashInferImpl.forward.

Cases (the probe gate from the work plan):
  bf16_d256          dense bf16, head 256, no SW
  bf16_d256_sw       dense bf16, head 256, sliding window 32 carried by the
                     custom mask (mm group planned window_left=-1)
  bf16_d512_vosplit  dense bf16, head_dim_qk=512 / head_dim_vo=256, two
                     VO-split passes per wrapper (production _run_vo_split_prefill)
  nvfp4_d256         uint8-packed NVFP4 KV + linear fp8 scale factors via the
                     vLLM `_fa2_nvfp4_prefill_jit_args` customize JIT module
                     (use_sliding_window=True module, mm plan window_left=-1,
                     SW carried by the mask: Gemma4 sliding-layer fidelity)
  bf16_d256_allmm    every request carries an in-window span (plain group
                     empty; one request with two spans)
  bf16_d256_nomm     spans exist in metadata but none intersect the query
                     window (span in computed context, decode-shaped qo=1,
                     degenerate start==end): the builder span filter must
                     classify all requests plain and the result must be
                     byte-identical to the scalar-causal plan (gated)

Reference: torch fp32 attention per request over the dequantized KV with the
same composed mask. Gates: cosine >= 0.9999 (bf16) / 0.999 (nvfp4), plus
plain-request rows must be torch.equal to a conventional full-batch causal
plan (scalar-causal regression), and mm rows must differ from pure-causal
output (the mask demonstrably engages).

  source env.sh && python test_mm_prefix_mask.py \
      --flashinfer-source-root ~/flashinfer --case all \
      --output results/mm_prefix_mask.json
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

COSINE_GATE_BF16 = 0.9999
COSINE_GATE_NVFP4 = 0.999

# Spans are document positions, inclusive [start, end] (vLLM mm_req_doc_ranges
# convention: triton helper masks q/k in [start, end], valid iff start < end).
# req2 has computed context (kv > qo); its span sits fully inside the query
# window, as guaranteed in vLLM by the forced --disable-chunked-mm-input.
# kv lens hit last_page_len {1, 16, 8, 13} at page_size 16.
REQUESTS = [
    dict(name="req0_text", qo_len=33, kv_len=33, spans=[]),
    dict(name="req1_mm", qo_len=48, kv_len=48, spans=[(8, 23)]),
    dict(name="req2_mm_append", qo_len=24, kv_len=40, spans=[(20, 35)]),
    dict(name="req3_text_append", qo_len=17, kv_len=29, spans=[]),
]

# Topology variants (kernel config fixed at bf16 d256): every request mm
# (plain group empty), and spans present in metadata but NOT intersecting
# the query window (span fully in computed context / degenerate start==end:
# the vLLM _mm_prefix_prefill_spans filter must classify ALL requests plain
# and the grouped path must be byte-identical to the scalar-causal plan).
REQUESTS_ALLMM = [
    dict(name="req0_mm", qo_len=33, kv_len=33, spans=[(0, 15)]),
    dict(name="req1_mm", qo_len=48, kv_len=48, spans=[(8, 23), (30, 41)]),
    dict(name="req2_mm_append", qo_len=24, kv_len=40, spans=[(20, 35)]),
    dict(name="req3_mm_append", qo_len=17, kv_len=29, spans=[(14, 27)]),
]
REQUESTS_NOMM = [
    dict(name="req0_text", qo_len=33, kv_len=33, spans=[]),
    # span fully inside the computed context (e < kv_len - qo_len)
    dict(name="req1_span_in_ctx", qo_len=8, kv_len=40, spans=[(2, 17)]),
    # decode-shaped row (qo_len=1) with span in context
    dict(name="req2_decode_like", qo_len=1, kv_len=29, spans=[(4, 19)]),
    # degenerate start == end span is invalid (triton: start < end)
    dict(name="req3_degenerate", qo_len=17, kv_len=17, spans=[(5, 5)]),
]


def effective_spans(qo_len, kv_len, spans):
    """Mirror of vLLM FlashInferMetadataBuilder._mm_prefix_prefill_spans
    filtering: valid iff start < end, intersecting the query window
    [kv_len - qo_len, kv_len). Keep in sync with flashinfer.py."""
    context_len = kv_len - qo_len
    return [
        (s, e) for s, e in spans if s < e and e >= context_len and s < kv_len
    ]

E2M1_TO_FLOAT32 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _configure_flashinfer_source_tree(source_root: Path | None) -> dict:
    """Point FlashInfer JIT at a source checkout instead of packaged data
    files (same shim as test_causal_grouping.py)."""
    if source_root is None:
        return {}
    source_root = source_root.expanduser().resolve()
    from flashinfer.jit import env as jit_env  # type: ignore

    jit_env.FLASHINFER_DATA = source_root
    jit_env.FLASHINFER_CSRC_DIR = source_root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = source_root / "include"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cutlass" / "include",
        source_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cccl" / "cub",
        source_root / "3rdparty" / "cccl" / "libcudacxx" / "include",
        source_root / "3rdparty" / "cccl" / "thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = source_root / "3rdparty" / "spdlog" / "include"
    info = {"source_root": str(source_root)}
    try:
        info["git_head"] = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        info["git_dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except Exception:
        pass
    return info


def build_mm_prefix_mask(torch, qo_len, kv_len, spans, sliding_window, device):
    """(causal AND sliding-window) OR span-bidirectional, end-aligned query
    window. sliding_window <= 0 disables SW. Mirrors the mask the vLLM
    FlashInfer builder will emit (and triton's compute_kv_seq_mask)."""
    q_abs = torch.arange(kv_len - qo_len, kv_len, device=device)
    k_abs = torch.arange(kv_len, device=device)
    mask = k_abs[None, :] <= q_abs[:, None]
    if sliding_window > 0:
        mask &= (q_abs[:, None] - k_abs[None, :]) < sliding_window
    for start, end in spans:
        if not (start < end):
            continue
        q_in = (q_abs >= start) & (q_abs <= end)
        k_in = (k_abs >= start) & (k_abs <= end)
        mask |= q_in[:, None] & k_in[None, :]
    return mask


def _torch_reference(torch, q, k_seq, v_seq, mask, sm_scale):
    """fp32 attention with an arbitrary qo_len x kv_len bool mask."""
    group = q.shape[1] // k_seq.shape[1]
    kf = k_seq.float().repeat_interleave(group, dim=1)
    vf = v_seq.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * sm_scale
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), vf)


def _make_nvfp4_kv(torch, shape, device):
    """Random packed-e2m1 bytes + linear fp8 scale factors (one per 16
    elements = 8 packed bytes). Same construction as
    scripts/flashinfer_nvfp4_kv_probe.py with global_scale 1.0."""
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    sf_shape = (*shape[:-1], shape[-1] // 8)
    choices = torch.tensor([56, 48, 40, 32], dtype=torch.uint8, device=device)
    sf = choices[torch.randint(0, 4, sf_shape, device=device)]
    return packed, sf.view(torch.float8_e4m3fn)


def _dequant_nvfp4(torch, packed, sf):
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    idx = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    lut = torch.tensor(E2M1_TO_FLOAT32, device=packed.device, dtype=torch.float32)
    values = lut[idx.to(torch.long)]
    return values * sf.to(torch.float32).repeat_interleave(16, dim=-1)


def _vllm_nvfp4_jit_args(torch, head_dim_qk, head_dim_vo, use_sliding_window):
    """Verbatim mirror of vllm _fa2_nvfp4_prefill_jit_args (bf16 q/o)."""
    uri = (
        "vllm_batch_prefill_nvfp4_kv_"
        "dtype_q_bfloat16_dtype_kv_fp4x2_e2m1_dtype_o_bfloat16_"
        "dtype_idx_int32_"
        f"head_dim_qk_{head_dim_qk}_head_dim_vo_{head_dim_vo}_"
        f"posenc_0_swa_{int(use_sliding_window)}_logits_cap_0_fp16_qk_0"
    )
    jit_args = [
        uri,
        torch.bfloat16,
        torch.uint8,
        torch.bfloat16,
        torch.int32,
        head_dim_qk,
        head_dim_vo,
        [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
            "maybe_k_cache_sf",
            "maybe_v_cache_sf",
        ],
        [
            "uint8_t", "int32_t", "float", "uint32_t",
            "uint16_t", "uint16_t", "uint8_t", "uint8_t",
        ],
        [
            "logits_soft_cap", "sm_scale", "rope_rcp_scale",
            "rope_rcp_theta", "token_pos_in_items_len",
        ],
        ["double", "double", "double", "double", "int64_t"],
        (
            "DefaultAttention<use_custom_mask, "
            f"{str(use_sliding_window).lower()}, false, false>"
        ),
        "#include<flashinfer/attention/variants.cuh>",
    ]
    jit_kwargs = {
        "pos_encoding_mode": 0,
        "use_sliding_window": use_sliding_window,
        "use_logits_soft_cap": False,
        "use_fp16_qk_reduction": False,
        "fp8_enabled": False,
    }
    return jit_args, jit_kwargs


def run_case(case_name: str, workspace_mb: int) -> dict:
    import torch

    import flashinfer

    device = "cuda"
    torch.manual_seed(0)

    if case_name == "bf16_d256_allmm":
        requests = REQUESTS_ALLMM
    elif case_name == "bf16_d256_nomm":
        requests = REQUESTS_NOMM
    else:
        requests = REQUESTS
    expect_byte_identical = case_name == "bf16_d256_nomm"

    nvfp4 = case_name == "nvfp4_d256"
    sliding_window = 32 if case_name in ("bf16_d256_sw", "nvfp4_d256") else -1
    if case_name == "bf16_d512_vosplit":
        head_dim_qk, head_dim_vo = 512, 256
    else:
        head_dim_qk, head_dim_vo = 256, 256
    num_splits = head_dim_qk // head_dim_vo
    num_qo_heads, num_kv_heads, page_size = 8, 4, 16
    sm_scale = head_dim_qk ** -0.5
    dtype = torch.bfloat16

    rec: dict = dict(
        case=case_name,
        nvfp4=nvfp4,
        sliding_window=sliding_window,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        requests=[dict(r) for r in requests],
    )

    # ---- synthetic batch + paged KV (pages round-robin interleaved) ----
    qo_lens = [r["qo_len"] for r in requests]
    kv_lens = [r["kv_len"] for r in requests]
    pages_per_req = [(kv + page_size - 1) // page_size for kv in kv_lens]
    total_qo = sum(qo_lens)
    page_ids = [
        [i + j * len(requests) for j in range(n)]
        for i, n in enumerate(pages_per_req)
    ]
    flat = sorted({p for ids in page_ids for p in ids})
    remap = {p: k for k, p in enumerate(flat)}
    page_ids = [[remap[p] for p in ids] for ids in page_ids]
    num_pages = len(flat)

    q = torch.randn(total_qo, num_qo_heads, head_dim_qk, dtype=dtype, device=device)

    k_seqs, v_seqs = [], []  # dequantized / dense reference sequences
    if nvfp4:
        kd_packed = head_dim_qk // 2
        k_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, kd_packed,
            dtype=torch.uint8, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        k_sf = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim_qk // 16,
            dtype=torch.float8_e4m3fn, device=device,
        )
        v_sf = torch.zeros_like(k_sf)
        for ids, kv_len in zip(page_ids, kv_lens):
            kp, ksf = _make_nvfp4_kv(
                torch, (kv_len, num_kv_heads, kd_packed), device
            )
            vp, vsf = _make_nvfp4_kv(
                torch, (kv_len, num_kv_heads, kd_packed), device
            )
            k_seqs.append(_dequant_nvfp4(torch, kp, ksf).to(dtype))
            v_seqs.append(_dequant_nvfp4(torch, vp, vsf).to(dtype))
            for j, page in enumerate(ids):
                n = min(page_size, kv_len - j * page_size)
                sl = slice(j * page_size, j * page_size + n)
                k_cache[page, :n] = kp[sl]
                v_cache[page, :n] = vp[sl]
                k_sf[page, :n] = ksf[sl]
                v_sf[page, :n] = vsf[sl]
    else:
        k_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim_qk,
            dtype=dtype, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        for ids, kv_len in zip(page_ids, kv_lens):
            k_seq = torch.randn(
                kv_len, num_kv_heads, head_dim_qk, dtype=dtype, device=device
            )
            v_seq = torch.randn_like(k_seq)
            k_seqs.append(k_seq)
            v_seqs.append(v_seq)
            for j, page in enumerate(ids):
                n = min(page_size, kv_len - j * page_size)
                k_cache[page, :n] = k_seq[j * page_size : j * page_size + n]
                v_cache[page, :n] = v_seq[j * page_size : j * page_size + n]

    # Batch-level planning arrays, exactly what the vLLM builder sees (CPU).
    qo_indptr_cpu = torch.tensor(
        [0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32
    )
    kv_indptr_cpu = torch.tensor(
        [0] + list(torch.tensor(pages_per_req).cumsum(0)), dtype=torch.int32
    )
    paged_kv_indices = torch.tensor(
        [p for ids in page_ids for p in ids], dtype=torch.int32, device=device
    )
    last_page_len_cpu = torch.tensor(
        [kv - (n - 1) * page_size for kv, n in zip(kv_lens, pages_per_req)],
        dtype=torch.int32,
    )
    rec["last_page_len"] = last_page_len_cpu.tolist()

    workspace = torch.empty(
        workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device
    )

    def make_wrapper():
        if nvfp4:
            jit_args, jit_kwargs = _vllm_nvfp4_jit_args(
                torch, head_dim_qk, head_dim_vo, use_sliding_window=True
            )
            return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", backend="fa2",
                jit_args=jit_args, jit_kwargs=jit_kwargs,
            )
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend="fa2"
        )

    def plan_wrapper(wrapper, req_indices, *, causal, window_left, custom_mask):
        qo_lens_t = qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]
        kv_pages_t = kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]
        g_qo = torch.zeros(len(req_indices) + 1, dtype=torch.int32)
        torch.cumsum(qo_lens_t[req_indices], dim=0, out=g_qo[1:])
        g_kv = torch.zeros(len(req_indices) + 1, dtype=torch.int32)
        torch.cumsum(kv_pages_t[req_indices], dim=0, out=g_kv[1:])
        page_gather = torch.cat(
            [
                torch.arange(
                    int(kv_indptr_cpu[i]), int(kv_indptr_cpu[i + 1]),
                    dtype=torch.int64,
                )
                for i in req_indices
            ]
        )
        g_indices = torch.index_select(
            paged_kv_indices, 0, page_gather.to(device)
        )
        plan_kwargs = dict(
            head_dim_vo=head_dim_vo,
            q_data_type=dtype,
            kv_data_type=torch.uint8 if nvfp4 else dtype,
            causal=causal,
            custom_mask=custom_mask,
        )
        parity = dict(
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=0.0,
            o_data_type=dtype,
            fixed_split_size=-1,
            disable_split_kv=False,
        )
        plan_params = inspect.signature(wrapper.plan).parameters
        plan_kwargs.update({k: v for k, v in parity.items() if k in plan_params})
        wrapper.plan(
            g_qo,
            g_kv,
            g_indices,
            last_page_len_cpu[torch.tensor(req_indices)],
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            **plan_kwargs,
        )
        token_indices = torch.cat(
            [
                torch.arange(
                    int(qo_indptr_cpu[i]), int(qo_indptr_cpu[i + 1]),
                    dtype=torch.int64,
                )
                for i in req_indices
            ]
        ).to(device)
        return token_indices

    def run_wrapper(wrapper, token_indices, out):
        """VO-split-aware run + scatter (production _run_vo_split_prefill)."""
        group_query = torch.index_select(q, 0, token_indices)
        group_out = torch.empty(
            (group_query.shape[0], num_qo_heads, head_dim_qk),
            dtype=dtype, device=device,
        )
        if nvfp4:
            packed_step = head_dim_vo // 2
            sf_step = head_dim_vo // 16
        else:
            packed_step = head_dim_vo
            sf_step = 0
        for split in range(num_splits):
            v_half = v_cache.narrow(-1, split * packed_step, packed_step)
            out_i = torch.empty(
                (group_query.shape[0], num_qo_heads, head_dim_vo),
                dtype=dtype, device=device,
            )
            kv_sf = (
                (k_sf, v_sf.narrow(-1, split * sf_step, sf_step))
                if nvfp4
                else None
            )
            wrapper.run(
                group_query,
                (k_cache, v_half),
                k_scale=1.0,
                v_scale=1.0,
                out=out_i,
                kv_cache_sf=kv_sf,
            )
            group_out[..., split * head_dim_vo : (split + 1) * head_dim_vo].copy_(
                out_i
            )
        out.index_copy_(0, token_indices, group_out)

    # ---- mm-prefix grouping: plain causal group + custom-mask group,
    # classified by the same span filter as the vLLM builder ----
    eff_spans = [
        effective_spans(r["qo_len"], r["kv_len"], r["spans"]) for r in requests
    ]
    mm_reqs = [i for i, s in enumerate(eff_spans) if s]
    plain_reqs = [i for i, s in enumerate(eff_spans) if not s]
    rec["mm_reqs"] = mm_reqs
    rec["plain_reqs"] = plain_reqs
    rec["effective_spans"] = eff_spans
    if expect_byte_identical:
        assert not mm_reqs, (
            "span filter must classify every NOMM request as plain; got "
            f"mm_reqs={mm_reqs}"
        )

    masks = {}
    for i in mm_reqs + plain_reqs:
        r = requests[i]
        masks[i] = build_mm_prefix_mask(
            torch, r["qo_len"], r["kv_len"], eff_spans[i], sliding_window, device
        )

    out = torch.zeros(total_qo, num_qo_heads, head_dim_qk, dtype=dtype, device=device)

    if plain_reqs:
        plain_wrapper = make_wrapper()
        plain_tokens = plan_wrapper(
            plain_wrapper, plain_reqs, causal=True,
            window_left=sliding_window - 1 if sliding_window > 0 else -1,
            custom_mask=None,
        )
        run_wrapper(plain_wrapper, plain_tokens, out)
    if mm_reqs:
        mm_custom_mask = torch.cat([masks[i].reshape(-1) for i in mm_reqs])
        rec["custom_mask_bits"] = int(mm_custom_mask.numel())
        mm_wrapper = make_wrapper()
        mm_tokens = plan_wrapper(
            mm_wrapper, mm_reqs, causal=False, window_left=-1,
            custom_mask=mm_custom_mask,
        )
        run_wrapper(mm_wrapper, mm_tokens, out)

    # ---- scalar-causal regression: full batch on one causal wrapper ----
    causal_out = torch.zeros_like(out)
    causal_wrapper = make_wrapper()
    causal_tokens = plan_wrapper(
        causal_wrapper, list(range(len(requests))), causal=True,
        window_left=sliding_window - 1 if sliding_window > 0 else -1,
        custom_mask=None,
    )
    run_wrapper(causal_wrapper, causal_tokens, causal_out)

    # ---- fp32 reference with the composed mask ----
    cosine_gate = COSINE_GATE_NVFP4 if nvfp4 else COSINE_GATE_BF16
    per_request = []
    all_ok = True
    for i, r in enumerate(requests):
        sl = slice(int(qo_indptr_cpu[i]), int(qo_indptr_cpu[i + 1]))
        ref = _torch_reference(
            torch, q[sl], k_seqs[i], v_seqs[i], masks[i], sm_scale
        )
        got = out[sl].float()
        of, rf = got.flatten(), ref.flatten()
        cosine = torch.nn.functional.cosine_similarity(of, rf, dim=0).item()
        max_abs = (of - rf).abs().max().item()
        finite = bool(torch.isfinite(got).all().item())
        is_mm = i in mm_reqs
        vs_causal_max_abs = (
            (out[sl].float() - causal_out[sl].float()).abs().max().item()
        )
        if is_mm:
            # the custom mask must demonstrably change the result vs causal
            mask_engaged = vs_causal_max_abs > 0
            regression_equal = None
        else:
            mask_engaged = None
            # Informational on mixed batches (split-kv scheduling may
            # differ between the partial and full plans); GATED on the
            # NOMM topology, where the grouped path degenerates to the
            # very same full-batch causal plan and must be byte-identical.
            regression_equal = bool(torch.equal(out[sl], causal_out[sl]))
        ok = finite and cosine >= cosine_gate and (mask_engaged is not False)
        if expect_byte_identical:
            ok = ok and regression_equal is True
        all_ok = all_ok and ok
        per_request.append(
            dict(
                request=r["name"],
                mm=is_mm,
                spans=r["spans"],
                qo_len=r["qo_len"],
                kv_len=r["kv_len"],
                cosine=cosine,
                max_abs_diff=max_abs,
                finite=finite,
                mask_engaged=mask_engaged,
                vs_causal_max_abs=vs_causal_max_abs,
                plain_rows_equal_causal_plan=regression_equal,
                ok=ok,
            )
        )
    rec["cosine_gate"] = cosine_gate
    rec["per_request"] = per_request
    rec["ok"] = all_ok

    hdr = (
        f"{'request':<20}{'mm':<5}{'qo':>4}{'kv':>4}{'cosine':>12}"
        f"{'max_abs':>10}{'engaged':>9}{'reg_eq':>8}  gate"
    )
    print(f"--- {case_name} ---")
    print(hdr)
    for row in per_request:
        print(
            f"{row['request']:<20}{str(row['mm']):<5}{row['qo_len']:>4}"
            f"{row['kv_len']:>4}{row['cosine']:>12.6f}{row['max_abs_diff']:>10.4f}"
            f"{str(row['mask_engaged']):>9}{str(row['plain_rows_equal_causal_plan']):>8}"
            f"  {'PASS' if row['ok'] else 'FAIL'}"
        )
    print(f"CASE {case_name}: {'PASS' if all_ok else 'FAIL'}")
    return rec


CASES = [
    "bf16_d256",
    "bf16_d256_sw",
    "bf16_d512_vosplit",
    "nvfp4_d256",
    "bf16_d256_allmm",
    "bf16_d256_nomm",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--flashinfer-source-root", type=Path, default=None)
    parser.add_argument("--case", choices=CASES + ["all"], default="all")
    parser.add_argument("--workspace-mb", type=int, default=128)
    args = parser.parse_args()

    record: dict = {
        "schema": "mm-prefix-custom-mask-probe/v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "env": {
            k: os.environ.get(k, "")
            for k in ("FLASHINFER_EXTRA_CUDAFLAGS", "PYTHONPATH")
        },
    }
    try:
        if args.flashinfer_source_root is not None:
            record["flashinfer_source_paths"] = _configure_flashinfer_source_tree(
                args.flashinfer_source_root
            )
        import torch

        import flashinfer

        record["torch"] = torch.__version__
        record["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
        record["device"] = torch.cuda.get_device_name(0)
        record["capability"] = list(torch.cuda.get_device_capability(0))

        cases = CASES if args.case == "all" else [args.case]
        record["cases"] = {}
        overall = True
        for case in cases:
            try:
                rec = run_case(case, args.workspace_mb)
            except Exception as exc:  # noqa: BLE001 - the failure IS the result
                rec = {
                    "case": case,
                    "ok": False,
                    "error": "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip(),
                    "error_traceback_tail": traceback.format_exc()[-3000:],
                }
                print(f"CASE {case}: FAIL ({rec['error']})")
            record["cases"][case] = rec
            overall = overall and bool(rec.get("ok"))
        record["ok"] = overall
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    except Exception as exc:  # noqa: BLE001
        record["ok"] = False
        record["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        record["error_traceback_tail"] = traceback.format_exc()[-3000:]

    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
