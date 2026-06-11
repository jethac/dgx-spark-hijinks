# Claude /goal, epoch 2 (rev 2, 2026-06-11)

Branch: `epoch2` (single working branch; pull --rebase before push).
Plan: docs/CAMPAIGN_PLAN_EPOCH2.md. Mail: mail/ per mail/README.md - check
at session start, stop points, box windows; coordinate with Codex by mail.

OPERATING MODE: autonomous. Work the queue continuously; do not stop to ask
for go-aheads between items. Stop ONLY when blocked on something only Jetha
can do (account access, hardware power, purchases, publication decisions)
or a genuine scope change. Report progress as work lands, not as requests.

PLATFORM POLICY (rev 2): Colab lane is PAUSED - do not iterate on the
notebook or the sm_120a wheel unless Jetha reopens it. sm_120 verification
runs on the LOCAL P520 (RTX 5060 Ti via WSL2 Ubuntu; rerunnable bring-up at
B:\workshop\wsl_sm120\, results pattern in
results/wsl_sm120_fix_validation_20260611/). The P520 is Claude-driven,
zero-contention; use it for all probe/kernel-level validation before any
Spark window. Spark = serving/capacity rows only, via marker protocol at
Codex's gaps.

QUEUE (work top to bottom; interleave Spark windows opportunistically):
1. DG-0 window packet (author offline): docker-pinned baseline per
   docs/DG0_SERVING_STACK_RECON.md + task 24; combined with the 31B bf16
   anchor row (r9 image, task 17) in ONE Spark window at Codex's next gap.
2. Run that window -> close the epoch-1 quality table + open the
   DiffusionGemma ledger. Mail Codex results.
3. FlashInfer canvas-mask enablement (bidirectional-in-canvas + causal
   prefix; generalizes the mm-prefix custom-mask work) - author + validate
   on the P520.
4. Split-dtype module keying (task 22) - author + P520-validate; mail Codex
   to unpark its graph gate.
5. DG-1 cache analysis -> DG-2 (full NVFP4 KV on DiffusionGemma) -> DG-3
   (KV-read-amplification benchmark).
6. Epoch-1 ladder debt as windows allow: E4B/12B AFTER rows, fp8 1-byte
   guard term, M4 (low).
Standing: keep docs/RESULTS_LEDGER.md and the support matrix current as
rungs flip; provenance gates (latch diag, EXT_PATH, md5) on every run; blog
remains gated on the full ladder.
