#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE_GENERATION="${IMAGE_GENERATION:-r9}"
export VLLM_SRC="${VLLM_SRC:-/home/jethac/spark_tmp/vllm-022-ad2337814-clone}"
export EXPECTED_VLLM_REF="${EXPECTED_VLLM_REF:-9759e3b06baa85db93e10ecc0a8afdc4199f449b}"
export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/jethac/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-76af798243d11c4910eaceaf1d62ba4227656d4a}"

exec bash "${SCRIPT_DIR}/build_vllm_gemma4_rebuiltc_r8_image.sh" "$@"
