# Claude-lane handover — 2026-06-12

You are taking over the **vLLM + FlashInfer lane** of the dgx-spark-hijinks
campaign (NVFP4 KV cache for Gemma on consumer/embedded Blackwell, CC 12.x)
from the Claude instance, which is pausing for a few days. The OTHER lane
(SGLang + images/infra) is run by a separate Codex instance — you coordinate
with it by committed mail files, never by assuming its state.

Read order for full context: this doc → docs/OVERNIGHT_LADDER_PLAN_20260612.md
(incl. all amendments + protocol patches) → docs/TRITON_RETIREMENT_SCORECARD.md
(criteria + adjudication log) → docs/RESULTS_LEDGER.md → mail/ (newest ~10).

## 1. Non-negotiable ground rules (Jetha's standing directives)

- ZERO-BUG BAR: "we cannot afford even the most minor bugs." Every PPL claim
  cell runs TWICE, bitwise-identical or the row is RED. Quantized-vs-bf16
  |delta| > 0.5 nats = RED pending investigation. Smoke transcripts banked
  verbatim. A RED with a verbatim error is a valid deliverable; a wrong GREEN
  is the disaster. Claim rows only on baked images / fully-verified installs.
- PROVENANCE GATES on every run: EXT_PATH import proof, NVFP4 linear-latch
  diagnostic, corpora md5s (c1 abb63f0e65247a25f870d3f2d57563ff,
  c2 1686a33b93ca17d1ecc6898d7d021781, c3 28dfeba997756c52a74ee74854411c4b),
  binary md5/image ids in status files. fp8 rows additionally need
  BOOT-PROFILE provenance (see §4 bistability). Order rows bf16 → nvfp4 → fp8.
- "Do it all": full Gemma 3 + 4 ladder, both engines, + DiffusionGemma
  DG rungs + Triton retirement + MTP + multimodal (image AND audio for
  E2B/E4B/12B). Blog/Colab-publication/upstream-filing are GATED on the
  completed ladder. llama.cpp NVFP4-KV contribution is a post-ladder capstone.
- All test checkpoints are -it variants. Spark memory guardrails: ONE server
  at a time, --gpu-memory-utilization 0.72, docker --memory 100g
  --memory-swap 100g.

## 2. Coordination protocols

- Repo: jethac/dgx-spark-hijinks branch `epoch2` — BOTH lanes push here.
  ALWAYS `git pull --rebase origin epoch2` before pushing.
- Mail: `mail/NNNN_<from>-to-<to>_<slug>.md`. `ls mail/` IMMEDIATELY before
  numbering, use max+1 (three collisions happened; the rule is load-bearing).
- Spark marker: `~/CLAUDE_WINDOW_OPEN` on the Spark host. Claim WRITE-FIRST
  (write marker, then verify docker ps empty; if docker busy, remove your
  marker and wait). Marker PERSISTS across all phases of one window — never
  clear between phases; append phase transitions to the marker file content.
  Clear at final release and VERIFY with ls. Marker present + docker empty
  >15 min = stalled holder: mail and self-clear. >3h-old marker + empty
  docker = stale.
- Spark host: 100.113.98.11 (tailnet), user jethac. Get the password from
  Jetha — NEVER write it into any file.

## 3. Platforms

- **Spark / GB10** (sm_121, 119 GiB unified): serving/capacity/claim rows.
  Corpora staged at /home/jethac/spark_tmp/claude_token_strat_20260612/docs/.
  Images: r9 `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
  (id 8c37bdbc4fdb) and r10 `...-sm121a-r10` (id aed0da3f96b2) = r9 +
  transformers 5.11.0 baked (fixes `gemma4_unified` unknown-arch crash;
  builder: scripts/build_vllm_gemma4_rebuiltc_r10_image.sh). PPL harness:
  scripts/vllm_prompt_ppl_sweep.py (--dump-token-logprobs exists), runner
  patterns in results/claude_*/run_*.sh. Known transient: first row after a
  window transition can hit an inductor-autotune "operation not permitted"
  crash — retry once before diagnosing.
- **P520** (this machine's WSL2 Ubuntu; RTX 5060 Ti, sm_120, 16GB): kernel
  probes + small-model serving. Drive via
  `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc '...'`. Envs:
  `~/sm120env` (torch 2.12.0+cu130) with editable `~/vllm-hijinks`
  (= 022 branch @ 9759e3b06, validated); `~/vllm_e2_env` + `~/vllm-e2`
  (e2-vllm branch; build may or may not have finished — check). FlashInfer
  source at `~/flashinfer` @ 7d5d477b (gitignored data/ symlinks were once
  missing — if JIT dies on a missing module, check those five symlinks).
  GPU is shared: check nvidia-smi, never kill others' processes.
  Housekeeping owed: WSL-internal `pip cache purge` (14G) once no build is
  running; C: drive was 100% full (caused 2 WSL crashes) — now ~48G free.
- **Colab G4 lane** (reopened): GCP G4 runtime = RTX PRO 6000, sm_120, 96GB.
  Notebook notebooks/colab_g4_gemma4_test_drive.ipynb (version KANGAROO;
  bump animal alphabetically in BOTH title banner and NOTEBOOK_VERSION on
  every push; mirror to slash-free `colab` branch). CI wheel workflow
  `.github/workflows/build-sm120a-wheel.yml` on jethac/vllm — STATE AT
  HANDOVER: run 1 (27382718191) FAILED at the "Require wheel" gate (the
  build step is continue-on-error with an always-run ccache save, so a
  partial compile still banks its cache); run 2 (27382746557) was IN
  PROGRESS on the restored cache and may simply complete. If run 2 also
  fails: `gh run view 27382746557 --repo jethac/vllm --log-failed` — the
  real compile error will be in the build step log (run 1's failed-step
  list was empty precisely because the gate, not the build step, "failed").
  Re-runs are cheap after the first cache save. CONFIRM the eventual
  release tag matches the notebook's `WHEEL_RELEASE_TAG =
  'sm120a-wheels-4e9f2ae9c'`. docs/COLAB_G4_LANE.md.

## 4. State of knowledge (what is PROVEN, with one-line provenance)

- **Triton retirement (Gemma 4, text-only, CC 12.x): VALIDATED + default-ON**
  on jethac/vllm `spark/hijinks-e2-vllm` @ 4e9f2ae9c. Paired cells all sizes
  PASS (E4B +0.004, 12B +0.028 on clean r10, 26B +0.009, 31B −0.040 = FI
  better; Triton tax 4.65317496471429 bitwise-confirmed). Speed parity,
  TTFT −6..−11%. Escape hatch VLLM_FLASHINFER_BF16_GEMMA=0. Knob can't leak
  to fp8/nvfp4/non-12.x/mm (74-cell test matrix).
  docs/TRITON_RETIREMENT_SCORECARD.md + results/claude_retirement_scorecard_20260612/.
- **Gemma 3 retirement: DELIBERATELY SCOPED OUT** — see bug below.
- **OPEN BUG (top priority for this lane):
  docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md** — FlashInfer serving
  numerics wrong at Gemma 3 1B geometry (d256 / SWA-512 / 1 KV head),
  observed on sm_120: FI-bf16 +0.221/+1.243/+1.380 nats vs two agreeing
  truth references; FI-nvfp4 deterministic gibberish on virgin JIT cache.
  Short prompts look fine (they never cross the 512 window). sm_121 NOT yet
  tested at this geometry. Bisect plan in the doc: (1) 1B rerun on Spark
  r10 (cheap — splits geometry vs platform), (2) logit-diff probe FI vs
  FLASH_ATTN on P520, (3) window-512/kv-heads-1 ablation. Forensic JIT
  cache preserved at WSL ~/.cache/flashinfer_prediag_070355.
- **fp8 per-boot BISTABILITY** (anomaly investigation, task #25): fp8 31B
  lands per-boot in one of TWO complete bitwise corpus profiles
  (A 4.4739/5.8363/3.0063, B 4.5915/5.8044/2.9636); request order does NOT
  select (the order-dependence theory was tested and REFUTED). fp8 beats
  bf16 on all corpora in BOTH profiles, so the anomaly claim survives with
  a profile-dependent margin. bf16: 6 bitwise repins; nvfp4: bitwise stable
  across 3 boots. OPEN: mechanism (per-boot autotune-class suspect).
  results/claude_order_control_20260612/ORDER_CONTROL_SUMMARY.md.
- **Anomaly strata**: fp8 = H-hard benign; nvfp4 prose inversion (+0.253 C2)
  = H-broad + secondary H-late, NOT a catastrophic-token bug. llama.cpp
  independent arm: direction replicates, magnitude ~10x smaller (stack-
  specific component unlocalized). results/claude_token_strat_20260612/.
- **MTP**: Gemma 4 ships -it-assistant KV-sharing drafters (all sizes);
  vLLM upstream support is first-class; drafter D512 pin bug FIXED (shared
  gemma4_global_attn_backend_override, merged). Gemma 3 has NO MTP assets —
  draft_model with gemma-3-270m-it. Identity gate: greedy spec output ==
  non-spec greedy, byte-identical. docs/MTP_DRAFTER_NOTES.md (§7 = Spark spec).
- **Audio**: Gemma 4 audio tokens are STRICTLY CAUSAL on all LM layers
  (image/video only get bidirectional spans) — our mask path correct by
  construction; 18 policy tests on branch `spark/hijinks-e2-audio` @
  7e326fd037 (tests-only; must merge with mm-retire). docs/AUDIO_MM_NOTES.md.
- **Ladder greens** (capacity ×, vs bf16): G3-12B 3.19, G4-12B 3.53,
  G4-26B 3.57, G4-31B ~3.5 (1.70 vs fp8), E4B SGLang 3.57, G3-1B fp8 2.0
  (nvfp4 RED per the bug). Full tables: docs/RESULTS_LEDGER.md.

## 5. Immediate work queue (in priority order)

1. **Check the mm-retire workstream** (was IN FLIGHT at handover): the
   flip implementation + tests are SAFE on pushed branch
   `spark/hijinks-e2-mm-retire` @ 9766264. The P520 build runs DETACHED
   under WSL systemd via `~/b_e2_supervisor.sh` (log `~/b_e2_supervisor.log`)
   and CONTINUES after the Claude session pauses - HARVEST it, don't
   restart: check the supervisor log first, then `~/vllm_e2_build_20260612.log`,
   then whether the supervisor proceeded into the smokes. The planned smokes:
   image-grounded mm serving (Gemma 3 4B-it + Gemma 4 E4B-it × bf16/nvfp4, FlashInfer
   mm route vs Triton route) from branch `spark/hijinks-e2-mm-retire`
   (UNMERGED — merge gate = smokes green). Look for results in
   results/p520_mm_retirement_smokes_20260612/, ~/mm_smoke_20260612/, and
   B:\workshop\agent_bus\status\. If unfinished: the harness conventions
   are in MM_PREFIX_MASK_NOTES; finish the smokes, then merge mm-retire +
   the audio tests branch into e2-vllm.
2. **Audio cells** (staged, one command): `bash /mnt/b/workshop/wsl_sm120/run_audio_mm_cells.sh`
   (port 8078) after the image smokes — E2B/E4B × routes × bf16/nvfp4.
3. **MTP identity ladder** (staged): B:\workshop\wsl_sm120\p520_mtp_identity_ladder.sh
   — G3 1B+270m, G4 E2B+assistant, bf16 + nvfp4, identity gate.
4. **P520 small-size text rows**: G3 4B + G4 E2B (vLLM) — reuse ~/vllm-e2
   install, bf16/nvfp4/fp8 with full gates; ADD bf16 Triton-comparator +
   on-box speed pairs (scorecard I1 for small sizes).
5. **1B-geometry bug bisect** (Spark, cheap): Gemma 3 1B on r10, FLASH_ATTN
   row vs FlashInfer row, C1 ctx 8191 — decides geometry-vs-platform.
   Then the P520 logit-diff probe per the bug doc.
6. **Spark mm + 12B-audio rows** from a post-merge image (bake from e2-vllm
   head AFTER step 1's merge; r10 builder script is the template).
7. **fp8 bistability mechanism** + stack-specific anomaly magnitude — design
   work; coordinate with Codex (its SGLang fp8 rows need profile provenance
   too, mail 0048).
8. Colab: confirm wheel tag when CI lands; Jetha test-drives KANGAROO.
9. DG-2 serving smoke on the dgemma branch (head dfb427952b) — was waiting
   on a Codex image (r10-class with DG support); check mail for its state.
10. Post-ladder (gated): upstream filing package (scorecard tables + bug
    docs + module-cache issue are the payload), llama.cpp NVFP4-KV capstone,
    blog finalization (B:\jethac.github.io\...2026-06-XX-nvfp4-kv-cache-support.md
    is current as of this morning incl. retirement/MTP/bug/bistability;
    Jetha owns frontmatter date/category + two prose nits).

## 6. Conventions and traps (hard-won; do not relearn)

- Windows host, bash with Unix syntax; Python file IO ALWAYS encoding='utf-8';
  prefer writing scripts to files over heredocs (heredocs have eaten multiple
  sessions); ast.parse-validate notebook cells before committing.
- Overlays must be Python-only with realpath-verified symlinks; never trust
  a result without EXT_PATH + latch proof; FlashInfer's JIT module cache is
  blind to EXTRA_CUDAFLAGS (cache-shadowing produced false greens twice —
  the r7 post-mortem).
- vLLM worktrees live under B:\workshop\worktrees\vllm\* (several); NEVER
  switch branches in a worktree you didn't create — make a new one.
- Wholesale-deterministic over minimal-iterative for env setup (Colab saga
  lesson). vLLM source builds on P520 take hours and compile fixed-arch
  kernels you can't skip — avoid; use the CI wheels or existing installs.
- Commit messages: end with your own agent attribution line (the existing
  history uses "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" for
  Claude-lane commits; use your own identity, keep the convention).
- B:\workshop\agent_bus\ was the Claude instance's subagent coordination
  bus (inbox/status/GPU-queue files). If your harness runs subagents, adopt
  it; otherwise ignore inbox/, but DO respect p520_gpu_queue.md claims.

## 7. Session-task snapshot (campaign tracker state at handover)

Open: #20 M4 page-unification bug (low), #25 anomaly (bistability mechanism
+ magnitude localization remain), #27 DG-2 smoke, #28 llama.cpp capstone
(gated), #29/#30 parked ideas (MSA sparse-attention crossover; extracting
"the consumer-Blackwell attention platform"), #31 mm masking (smokes+merge
remain), #33 retirement (mm/audio closure + filing remain), #34 MTP
(identity ladder + Spark spec remain), #35 Colab G4 (wheel-tag confirm).
Done and banked: dispatcher fix, VO-split, split-dtype, DG-0/DG-1/DG-2 code,
corpus sweep, token strata, llama.cpp arm, order control, scorecard, r10,
G4-12B closure, 1B verification, audio policy, MTP enablement, CI+notebook.
