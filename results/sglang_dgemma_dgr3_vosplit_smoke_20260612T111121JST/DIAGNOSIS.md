# DG-R3 VO-Split Smoke First Run Diagnosis

Date: 2026-06-12 JST

Status: RED, but not a kernel crash.

What happened:

- Server reached readiness and answered all three revised DG-R2 text prompts
  with semantically correct non-empty responses.
- The strict revised gate failed because the DGX Spark prompt was not
  byte-stable across two repeats.
- The packet omitted the deterministic `Gemma4Renoise` config used by the DG-R2
  revised green row (`seed: 1234`, `max_denoising_steps: 48`, same sampler and
  stopping thresholds).
- The generated summary also missed valid routing proof because the live
  geometry trace records split passes as `extend_paged_vosplit0` /
  `extend_paged_vosplit1`; the inner split call logs `vo_split=False`.

Important evidence:

- Policy opt-in warning is present in `server.log`.
- D=512 layers route through split-pass geometry lines with
  `head_dim_vo=256`, for example layer 5 and layer 23.
- `docker_ps_after.txt` is empty; the server was torn down.

Fix staged after this row:

- `scripts/run_sglang_dgemma_dgr3_vosplit_smoke.sh` now writes and passes the
  deterministic `dllm_config.yaml`.
- The route parser now accepts D=512 `*_vosplit0/1` trace labels with
  `head_dim_vo=256`.
