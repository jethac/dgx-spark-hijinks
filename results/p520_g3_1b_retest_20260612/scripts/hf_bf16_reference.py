#!/usr/bin/env python3
"""HF transformers bf16 eager C1 ground-truth PPL for Gemma 3 1B.

Scores the SAME ctx=8191 token window the vLLM PPL harness scores (positions
1..N-1, mean NLL in nats), so the reference is directly comparable to the
served rows. Eager attention, bf16, single forward pass.
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-3-1b-it"
CORPUS = "/home/jetha/corpora/hijinks_serving/c1_ppl_corpus.md"
CTX = 8191
OUT = sys.argv[1] if len(sys.argv) > 1 else "hf_bf16_reference_ppl.json"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    text = open(CORPUS, encoding="utf-8", errors="replace").read()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < CTX:
        raise ValueError(f"corpus tokenized to {len(ids)} < ctx {CTX}")
    ids = ids[:CTX]
    input_ids = torch.tensor([ids], dtype=torch.long)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).cuda().eval()

    with torch.no_grad():
        input_ids = input_ids.cuda()
        out = model(input_ids=input_ids)
        logits = out.logits.float()  # [1, N, V]
    # Predict position t from logits at t-1. Score positions 1..N-1 (matches
    # the vLLM harness which leaves position 0 unscored).
    shift_logits = logits[0, :-1, :]          # predicts tokens 1..N-1
    shift_targets = input_ids[0, 1:]          # tokens 1..N-1
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    tgt_lp = logprobs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
    mean_nll = float(-tgt_lp.mean().item())
    num_scored = int(shift_targets.numel())

    report = {
        "schema": "hf-bf16-reference-ppl/v1",
        "model": MODEL,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "results": {
            "c1": {
                "file": "c1_ppl_corpus.md",
                "ctx": CTX,
                "scored_tokens": num_scored,
                "mean_nll_nats": mean_nll,
            }
        },
    }
    open(OUT, "w", encoding="utf-8").write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
