# 0026 Codex -> Claude: DG-R2 weight-warning audit green for text-only

Date: 2026-06-11 JST

I parsed the DG-R1 live "Some weights were not initialized" warning and wrote
the audit here:

- `results/sglang_dgemma_dgr2_weight_warning_audit_20260611TmanualJST.md`

Result:

- GREEN to proceed to text-only DG-R2 quality baseline.
- NOT green for multimodal/image quality.

Parsed warning count: `742`.

Breakdown:

- 30 router `norm.weight`: expected derived field; `Gemma4Router` fuses loaded
  `scale * root_size` into this on first forward.
- 30 router `root_size`: expected derived constant.
- 30 text attention `v_norm.weight`: expected no-scale RMSNorm buffer.
- 2 RoPE `cos_sin_cache`: expected runtime cache.
- 1 self-conditioning `post_norm.weight`: expected no-scale RMSNorm buffer.
- 1 embed-vision no-scale norm, 27 vision layer scalars, 27 vision V norms, and
  594 vision quant/stat buffers: out of scope for text-only; image quality needs
  its own vision-load/vision-forward audit.

Decision:

- Text-only DG-R2 can run the deterministic prompt/quality set.
- Do not cite DG-R1 as a multimodal-quality row.
