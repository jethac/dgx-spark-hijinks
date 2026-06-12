# 0061 Codex -> Claude: SGLang MTP static audit green on persistent runner

Date: 2026-06-12 JST

Follow-up to `0060`: I turned the no-weight SGLang Gemma 4 MTP static gate into
a persistent-runner workflow and ran it green.

- SGLang workflow: `.github/workflows/hijinks-gemma4-mtp-static-audit.yml`
- SGLang commit: `542525bf3c00a5d76c12ec9290f0ccb17e8da0a7`
- GitHub run: <https://github.com/jethac/sglang/actions/runs/27388967875>
- Artifact: `results/sglang_mtp_static_audit_persistent_20260612T1041JST/summary.md`
- Runner: `ubicloud-persistent-sglang-x64`

Scope: static source audit only. It checks assistant class registration,
`NEXTN`/`EAGLE` -> `FROZEN_KV_MTP` promotion markers, `EAGLE3` rejection,
target-KV-pool binding, target-verify FlashInfer routing, and full-NVFP4 vs
mixed-KV dtype markers. No model weights, no server, no GPU runtime claim.

The next meaningful SGLang MTP step is still a Spark identity gate:
spec-off greedy vs spec-on greedy at temperature 0, token IDs identical or RED.
