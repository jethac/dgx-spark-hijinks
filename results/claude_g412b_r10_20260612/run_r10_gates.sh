#!/usr/bin/env bash
# r10 provenance gates - same gate set as r9
# (results/vllm_gemma4_rebuiltc_image_r9_verification_20260611.md):
#   1. GPU runtime import probe (vllm/flashinfer/torch/humming + extension
#      imports + CC/SMs) PLUS the r10-specific transformers 5.11.0 +
#      gemma4_unified check
#   2. cuobjdump -lelf on _C.abi3.so -> sm_121a cubins
#   3. scripts/nvfp4_linear_latch_diag.py -> "writer wrote LINEAR V-SF"
#   4. FlashInfer module-cache audit (no payload in /root/.cache or /tmp)
set -uo pipefail
R=/home/jethac/spark_tmp/claude_g412b_r10_20260612
IMAGE=${IMAGE:-jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10}
S=$R/status.txt
G=$R/results

echo "R10_GATES_START $(date -Is) IMAGE=${IMAGE} IMAGE_ID=$(docker images --format '{{.ID}}' ${IMAGE})" >> "$S"
docker image inspect "${IMAGE}" > "$G/r10_image_inspect.json"

# Gate 1: import probe (GPU)
docker run --rm --gpus all --net host --ipc host -w /work \
  -v "${R}:/work" "${IMAGE}" python3 - <<'PY' > "$G/r10_import_probe.txt" 2>&1
from pathlib import Path
import importlib.metadata as md
import torch
import vllm
import flashinfer
import humming
import transformers

print("vllm", getattr(vllm, "__version__", None), vllm.__file__)
print("flashinfer", getattr(flashinfer, "__version__", None), flashinfer.__file__)
print("torch", torch.__version__, torch.version.cuda)
print("humming", humming.__file__, md.version("humming-kernels"))
print("transformers", transformers.__version__, transformers.__file__)
assert transformers.__version__ == "5.11.0", transformers.__version__
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
except Exception:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as M
assert "gemma4_unified" in M, "gemma4_unified missing"
print("GEMMA4_UNIFIED present in transformers config mapping")
assert torch.cuda.is_available()
print("device", torch.cuda.get_device_name(0),
      "cc", torch.cuda.get_device_capability(0),
      "sms", torch.cuda.get_device_properties(0).multi_processor_count)
print("arch_list", torch.cuda.get_arch_list())
import vllm._C  # noqa
import vllm._C_stable_libtorch  # noqa
import vllm._moe_C  # noqa
import vllm.vllm_flash_attn._vllm_fa2_C  # noqa
print("extension imports: PASS")
PY
echo "GATE1_IMPORT_PROBE_RC=$?" >> "$S"

# Gate 2: sm_121a cubins
docker run --rm -w /work -v "${R}:/work" "${IMAGE}" bash -lc \
  'VLLM_C="$(python3 -c "from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent / \"_C.abi3.so\")")"; /usr/local/cuda/bin/cuobjdump -lelf "$VLLM_C" | grep -E "sm_121|sm_120|compute_121|compute_120" | sort | uniq -c' \
  > "$G/r10_cuobjdump_sm121.txt" 2>&1
echo "GATE2_CUOBJDUMP_RC=$? SM121A_LINES=$(grep -c sm_121a "$G/r10_cuobjdump_sm121.txt")" >> "$S"

# Gate 3: linear-latch diag (needs nvfp4_linear_latch_diag.py + roundtrip probe in $R/scripts)
docker run --rm --gpus all --net host --ipc host -w /work \
  -e VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  -v "${R}:/work" "${IMAGE}" \
  python3 scripts/nvfp4_linear_latch_diag.py --output results/r10_latch_diag.json \
  > "$G/r10_latch_diag_stdout.txt" 2>&1
echo "GATE3_LATCH_RC=$? VERDICT=$(python3 -c "import json; print(json.load(open('$G/r10_latch_diag.json')).get('verdict'))" 2>/dev/null)" >> "$S"

# Gate 4: module-cache audit
docker run --rm -w /work -v "${R}:/work" "${IMAGE}" bash -lc \
  'echo FLASHINFER_AOT_DIR=${FLASHINFER_AOT_DIR:-unset}; for d in /root/.cache/flashinfer /root/.cache/flashinfer-aiter /tmp/flashinfer /tmp/flashinfer_modules; do echo "== $d"; ls -laR "$d" 2>&1 | head -20; done; echo "== payload .so search"; find /root/.cache /tmp -name "*.so" 2>/dev/null | head' \
  > "$G/r10_module_cache_listing.txt" 2>&1
echo "GATE4_CACHE_AUDIT_RC=$? PAYLOAD_SO=$(grep -c '\.so$' "$G/r10_module_cache_listing.txt")" >> "$S"

echo "R10_GATES_DONE $(date -Is)" >> "$S"
