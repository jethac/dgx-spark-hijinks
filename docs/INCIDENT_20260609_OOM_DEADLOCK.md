# Incident 2026-06-09 — vLLM unified-memory OOM → kernel hung-task deadlock

Status: root-caused. **Not thermal.** A vLLM run exhausted the GB10 unified memory pool;
the global OOM-killer fired on the vLLM EngineCore while it held an mmap rw-semaphore inside
the NVIDIA driver, cascading into a system-wide hung-task deadlock. The box went catatonic
(SSH/TCP-22 dead, Tailscale one-way) and required a hard power-cycle. It is recurring.

## TL;DR
- **Cause:** global out-of-memory on the 119 GiB unified pool, triggered by a vLLM serving
  run (process `VLLM::EngineCor` mapped ~123 GB), running in a Docker container.
- **Why it wedged instead of just OOM-killing cleanly:** the killed process held an mmap
  rw-semaphore in the NVIDIA driver mid-allocation, so `jbd2` (ext4 journal), `gnome-shell`,
  `gsd-color`, `systemd-journal` all blocked on it → hung-task deadlock that never recovered.
- **Why it needed a physical power-cycle:** `kernel.hung_task_panic=0`, so the kernel did
  not auto-reboot on the hang — it sat frozen.
- **NOT thermals:** GPU 40 °C post-reboot; boot-time zones 39–41 °C; zero throttle/overtemp
  events anywhere near the incident.
- **Recurring:** NVRM out-of-memory + OOM-kills on Jun 06, Jun 07, and Jun 09 in a single
  boot — the serving runs have been repeatedly over-committing memory.

## Evidence
- OOM-killer named the culprit:
  ```
  oom-kill: ... global_oom, task_memcg=/system.slice/docker-722800…scope,
            task=VLLM::EngineCor, pid=400721
  ```
  `total_vm` 32,264,330 pages × 4 KiB ≈ **123 GB** on a **119 GiB** box.
- Lock cascade: `task gnome-shell blocked on an rw-semaphore likely owned by task
  VLLM::EngineCor (writer)`; `jbd2/nvme0n1p2`, `gsd-color`, `systemd-journal` blocked
  122 → 245 → 368 s and never cleared.
- Memory: `MemTotal 119 GiB`, `Swap 15 GiB` (swap is irrelevant — GPU-mapped memory is not
  swappable). `vm.overcommit_memory=0`, `vm.swappiness=60`, `kernel.hung_task_panic=0`,
  `kernel.hung_task_timeout_secs=120`.
- Thermal: current 40 °C; no throttle/overtemp/emergency lines. Heat exonerated.
- The Tailscale "online but rx=0 / one-way" state seen during the outage was earlier
  endpoint/NAT churn (~05:00–05:20 JST, normal magicsock reshuffling, no errors) — likely
  separate minor network flakiness, not the kill.

## Root cause — and why it's Spark-specific
GB10 has **unified memory**: CPU and GPU share the same ~119 GiB. vLLM's
`--gpu-memory-utilization` is a fraction of *that shared pool*, not a separate VRAM budget.
A 27B-class model + a large NVFP4 KV pool can exceed the pool, and the **matched
fp8-vs-nvfp4 comparator pattern is the obvious trigger** — two servers up at once doubles
the footprint. When the pool is exhausted: NVRM can't allocate (`NV_ERR_NO_MEMORY`) →
global OOM → kill `VLLM::EngineCor` → but it holds the driver mmap lock → everything blocks
→ deadlock. On a discrete-VRAM box a GPU-OOM just kills the CUDA process; on unified memory
it can take down the whole kernel.

## Standing mitigations (apply to all GB10 serving runs)
1. **Leave OS headroom.** Treat 119 GiB as shared. Keep ≥15–20 GiB free for OS/Docker/
   `jbd2`. Lower `--gpu-memory-utilization` accordingly; do not run at 0.85+ on this box.
2. **Do not run the fp8 + nvfp4 comparator servers concurrently** at high mem-fraction.
   Run them sequentially, or cap each so the SUM fits under ~100 GiB. Tear down server A
   before starting server B for big models.
3. **Cgroup-limit the container** (`docker run --memory=<N>g --memory-swap=<N>g`, or a
   systemd slice limit). A *cgroup* OOM kills the container's process predictably; the
   *global* OOM + driver-held mmap_lock is what deadlocks the kernel. This is the single
   best protection against a repeat hard-wedge.
4. **Make it self-recover** (headless box) — **APPLIED 2026-06-09** in
   `/etc/sysctl.d/99-spark-oom-recovery.conf`: `kernel.hung_task_panic=1`,
   `kernel.hung_task_timeout_secs=300`, `kernel.panic=30`. A genuine deadlock (task stuck
   > 5 min) now panics and auto-reboots ~5.5 min in, instead of staying catatonic until a
   physical power-cycle. Timeout raised from the 120 s default to 300 s so transient hangs
   don't trigger false reboots.
5. Swap is not the lever (GPU-mapped memory isn't swappable). Cap allocation, don't add swap.

## Pre-run check
Before a big serving run, confirm the pool is clear and size the run to fit:
```bash
free -h                      # expect ~115 GiB available before launch
nvidia-smi                   # confirm no stale process holding memory
# size: model_weights + KV_pool + ~20 GiB OS headroom <= 119 GiB
```
And after the host-access gate (`usable_for_live_work: true`), prefer one big server at a
time over concurrent comparators.
