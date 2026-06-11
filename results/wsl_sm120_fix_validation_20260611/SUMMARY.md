# Dispatcher fix VALIDATED on sm_120 (local 5060 Ti testbed) - 2026-06-11

Platform: RTX 5060 Ti (CC 12.0, driver 595.79), WSL2 Ubuntu 24.04, CUDA
13.0.88, torch 2.12.0+cu130, FlashInfer 76af7982 source-tree. Rerunnable
bring-up scripts in this dir (canonical copies: B:\workshop\wsl_sm120\).

| run | result | reference | match |
|---|---|---|---|
| rt-base (page32, tiny lens) | GREEN | was RED on fb7d62ea (GB10+sm120) | FIX CONFIRMED |
| rt5 (page16, tiny lens) | GREEN | was RED | FIX CONFIRMED |
| e4b/31b bf16 vo-split | PASS 0.9999979-80 | same cosines GB10/Colab | unchanged |
| nvfp4 A1 NHD/HND | PASS 0.9999985-86 | same | unchanged |
| fp8 (512,256) trait | RED, verbatim 1-byte-guard error | must stay red | not masked |

NEW STANDING PLATFORM: the P520/5060 Ti is the campaign's third GPU - the
always-on sm_120 probe/kernel testbed (75s first JIT, no Spark contention,
no Colab preemption). FlashInfer changes get same-hour validation here
before any Spark serving window.

Remaining for task 17: GB10 serving-level proof (31B bf16 anchor row) -
needs r9 image or source-mount on the Spark; the kernel-level fix is done.
