# 26B-A4B: top-logprob/readout attribution packet staged

Status: no live run yet; this is a prepared next discriminator after the completed e0 layer split.

I added:

- `docs/vast_anchor/vllm_toplogprob_attribution.py`
- `docs/vast_anchor/run_26b_toplogprob_attribution.sh`

Purpose: test the next open branch from the bug doc without rebuilding vLLM. It uses vLLM
`prompt_logprobs` as a final-readout proxy and compares bf16 against selected NVFP4 first-block rows at the
same supplied-token positions.

Default rows:

- `bf16`
- `base_k100`
- `e0_all`
- `l0`
- `l1`

Default capture:

- `prompt_logprobs=20`
- top-20 kept
- dense top-k capture for first 256 scored suffix positions
- stride-16 top-k capture through the remaining suffix
- full per-position target NLL summaries for all 4088 scored tokens

Outputs:

- `summary.tsv`
- `toplogprob_delta_report.tsv`
- per-row JSON with position summaries and sampled top-logprobs

What it should answer:

- If NVFP4 low-NLL rows have high bf16-vs-NVFP4 top-k overlap but target NLL shifts strongly, inspect
  supplied-token/logprob extraction or a narrow target-token effect.
- If top-k overlap/top-1 match collapses in `e0_all`/`l0`/`l1`, the first-block perturbation is visibly
  changing the final distribution, and the next heavier hook should capture final hidden/lm-head around the
  same positions.

Validation done locally: `python -m py_compile docs/vast_anchor/vllm_toplogprob_attribution.py`.

No Vast instance is active from this work.
