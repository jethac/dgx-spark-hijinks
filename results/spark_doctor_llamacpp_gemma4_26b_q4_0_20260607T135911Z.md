# Spark Doctor

## Findings
- WARN: PyTorch did not report an available CUDA device.
- OK: host architecture is ARM64/aarch64.
- OK: nvidia-smi query is consistent with GB10 / compute capability 12.1.

## Platform
- `system`: `Linux`
- `machine`: `aarch64`
- `processor`: `aarch64`
- `python`: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- `executable`: `/usr/bin/python3`

## Commands
### nvidia_smi
- returncode: `0`

```text
Sun Jun  7 23:00:07 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   41C    P0             10W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          329657      G   /usr/lib/xorg/Xorg                       43MiB |
|    0   N/A  N/A          329691      G   /usr/bin/gnome-shell                     16MiB |
|    0   N/A  N/A          454928      C   ...-b9536/build/bin/llama-server      15136MiB |
+-----------------------------------------------------------------------------------------+
```
### nvidia_smi_query
- returncode: `0`

```text
NVIDIA GB10, 12.1, 580.159.03, [N/A]
```
### nvcc
- returncode: `0`

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:57:39_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```
### cuobjdump
- returncode: `0`

```text
cuobjdump: NVIDIA (R) fat binary listing tool
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Thu_Aug_14_07:31:26_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.85
Build cuda_13.0.r13.0/compiler.36400806_0
```
### uname
- returncode: `0`

```text
Linux thinkstationpgx-00b4 6.17.0-1021-nvidia #21-Ubuntu SMP PREEMPT_DYNAMIC Wed May 27 19:14:05 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
```

## Python Modules
- `vllm`: not available
- `flashinfer`: not available
- `triton`: not available
- `transformers`: not available
- `lm_eval`: not available
- `llama_cpp`: not available
- `torch`: not available
