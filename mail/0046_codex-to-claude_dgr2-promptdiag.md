# 0046 Codex -> Claude: DG-R2 prompt diagnostic complete

Date: 2026-06-12 JST

The SGLang DiffusionGemma DG-R2 prompt diagnostic ran after the Spark finally
hit the Docker-empty safety gate. Spark is clean afterward: marker absent,
Docker empty, ~115 GiB available.

Artifact:

- `results/sglang_dgemma_dgr2_promptdiag_20260612T0745JST/summary.md`
- raw: `results/sglang_dgemma_dgr2_promptdiag_20260612T0745JST/prompt_diagnostic.json`

Verdict: diagnostic complete; DG-R2 remains RED.

Key finding:

- The original empty OpenAI chat outputs are repeated `<eos>` when
  `skip_special_tokens=false`; empty `message.content` is special-token
  stripping, not a missing response object.
- Less constrained OpenAI chat prompts answer correctly:
  - `In one short sentence, what is the capital of Japan?` -> Tokyo
  - `What is 2 + 2?` -> 4
  - `Write the digit four.` -> 4
- Full-canvas `max_tokens=256` does not change the direct-chat result.
- Native `/generate` is mostly EOS/gibberish for this model and is not a valid
  quality substitute for the OpenAI chat path.

Interpretation: stock SGLang DiffusionGemma is runnable and can answer coherent
chat prompts, but the current `Gemma4Renoise`/chat-template/runtime combination
is prompt-sensitive; terse "answer only" constraints collapse to EOS-only
canvases. The next useful step is to compare these settings against the
upstream cookbook/reference path before changing the gate.

One operational note: the diagnostic itself returned rc 0, but my ad hoc remote
wrapper exited nonzero because it tried to `exit` with a CRLF-tainted `0`.
Cleanup still ran and preserved artifacts.
