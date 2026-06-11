# vLLM Gemma4 Rebuilt-C Image Import Probe

Date: 2026-06-10 15:45 JST

Image: `jethac-vllm-aeon-gemma4:ad2337814-rebuiltc-fb7d62ea-sm121a`

Purpose: independently verify that the rebuilt image created in
`results/vllm_gemma4_rebuiltc_image_build_20260610T1440JST_r7_summary.md` imports the
rebuilt vLLM CUDA extensions after Docker export. This is an image/build receipt, not a
serving or model-quality result.

Command:

```bash
docker run --rm --gpus all \
  jethac-vllm-aeon-gemma4:ad2337814-rebuiltc-fb7d62ea-sm121a \
  python3 -c '... import vllm, flashinfer, torch; import vllm._C ...'
```

Output:

```text
vllm 0.1.dev1+gad2337814 /opt/jethac-vllm/vllm/__init__.py
flashinfer 0.6.13 /opt/jethac-flashinfer/flashinfer/__init__.py
torch 2.11.0+cu130 13.0
_C.abi3.so True 88005848
_C_stable_libtorch.abi3.so True 159411304
_moe_C.abi3.so True 78096104
vllm_flash_attn/_vllm_fa2_C.abi3.so True 229156064
vllm_flash_attn/_vllm_fa3_C.abi3.so True 907469760
imported vllm._C /opt/jethac-vllm/vllm/_C.abi3.so
imported vllm._moe_C /opt/jethac-vllm/vllm/_moe_C.abi3.so
imported vllm.vllm_flash_attn._vllm_fa2_C /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
```

Build-target receipt from the Docker build log:

```text
ELF file    1: _C.abi3.1.sm_121a.cubin
...
ELF file   30: _C.abi3.30.sm_121a.cubin
```

Scope:

- vLLM ref: `ad233781492ca1d4eaa8c1dd0d80777933163d54`
- FlashInfer ref: `fb7d62ea45f19cb61f19057a93519c17b6e257f3`
- Base image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- Image id: `sha256:750ce7b9c59befe4426b484be24a1f58f585a5e13b7ebe13179a16f4feb4d051`

Result: import/build green. No server was launched, and `docker ps` was empty after the
probe.
