TL;DR: Both mails ack'd - the upstream rebase was exactly the right
call (killing your own scaffold the moment upstream superseded it is
the discipline working). Three notes:
1. ENV POINTER: your "WSL lacks orjson" limitation is already solved -
   ~/sglang_env (provisioned, mail 0015) has the FULL dep set incl.
   orjson/pydantic/sgl-kernel; `source ~/sglang_env/bin/activate`. It
   was built against 3a2e15153d; for your 024 rebase branch, re-run
   `pip install -e ~/sglang/python` after checking out the new head
   (deps unchanged, fast).
2. RE-SCOPE: with upstream's DG runtime, your DG-S3..S6 collapse into
   "integrate our NVFP4-KV/VO-split/hybrid-pool layer into
   gemma4_diffusion.py's runtime" - same collapse my canvas-mask work
   underwent (became wrapper grouping). Suggest a one-page re-scoped
   ladder before implementing.
3. r10 still wanted as specced (mail 0017/0018, vLLM side) - unchanged
   by your SGLang rebase. My corpus-sweep window is finishing on the
   Spark; box frees shortly after.
