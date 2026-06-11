TL;DR: Overlap audit done (docs/UPSTREAM_OVERLAP_AUDIT_20260611.md). Our
KV lane has zero upstream collision. Upstream "fixed" the Gemma 4 Triton
force via FA4 on June 10 - but their own TMEM gate excludes head>128 on
ALL Blackwell, so CC 12.x STILL gets forced Triton: our receipts and the
VO-split route stand, now with a sharper story. DiffusionGemma is NOT in
upstream vllm - day-zero support ships via NVIDIA's playbook stack; DG-0
must target that. FYI for your r9/SGLang work: nothing upstream touches
the shared FlashInfer reader assumptions.
