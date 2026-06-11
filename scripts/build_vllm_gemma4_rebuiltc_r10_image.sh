#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_vllm_gemma4_rebuiltc_r10_image.sh [IMAGE_TAG]

r10 = the verified r9 image (vLLM 9759e3b06, FlashInfer 76af7982, sm_121a
rebuilt-C, id-pinned) + transformers pinned to 5.11.0, the version the
2026-06-12 retirement-scorecard overlay proved serves gemma4_unified
(G4-12B) green on vLLM 9759e3b06. Per the adjudication-log r10 spec
(docs/TRITON_RETIREMENT_SCORECARD.md): "r9 recipe + transformers >= the
version knowing gemma4_unified; same provenance gates".

Rather than re-running the multi-hour rebuilt-C compile, this derives r10
FROM the banked final r9 image after asserting its image id matches the
verified sha256 (results/vllm_gemma4_rebuiltc_image_r9_verification_20260611.md).
The transformers pin layer asserts at build time that the pinned version
imports and that `gemma4_unified` is present in the transformers config
mapping. FlashInfer module-cache hygiene is re-scrubbed in the final layer
(same dir set as the r8/r9 precacheclean step).

The r9-equivalent provenance gates (GPU import probe, sm_121a cubins,
linear-latch diag, module-cache audit) run POST-build on GB10; this script
is build + build-time asserts only.

Environment:
  BASE_IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
  EXPECTED_BASE_ID=sha256:8c37bdbc4fdb1cc6bef279ebac011362cf8a14033fcc739e65fb5e656d326eea
  TRANSFORMERS_PIN=5.11.0
  RESULTS_DIR=/home/jethac/dgx-spark-hijinks/results
  RUN_ID=vllm_gemma4_rebuiltc_image_build_YYYYMMDDTHHMMJST_r10
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

IMAGE_TAG=${1:-jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10}
BASE_IMAGE=${BASE_IMAGE:-jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9}
EXPECTED_BASE_ID=${EXPECTED_BASE_ID:-sha256:8c37bdbc4fdb1cc6bef279ebac011362cf8a14033fcc739e65fb5e656d326eea}
TRANSFORMERS_PIN=${TRANSFORMERS_PIN:-5.11.0}
IMAGE_GENERATION=r10
RUN_ID=${RUN_ID:-vllm_gemma4_rebuiltc_image_build_$(TZ=Asia/Tokyo date +%Y%m%dT%H%MJST)_${IMAGE_GENERATION}}
RESULTS_DIR=${RESULTS_DIR:-/home/jethac/dgx-spark-hijinks/results}
LOG_PATH=${RESULTS_DIR}/${RUN_ID}.log
SUMMARY_PATH=${RESULTS_DIR}/${RUN_ID}_summary.md

DOCKERFILE=$(mktemp)
BUILD_CONTEXT=$(mktemp -d)
trap 'rm -f "${DOCKERFILE}"; rm -rf "${BUILD_CONTEXT}"' EXIT
mkdir -p "${RESULTS_DIR}"

actual_base_id=$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE}")
if [[ "${actual_base_id}" != "${EXPECTED_BASE_ID}" ]]; then
  echo "wrong base image id: got ${actual_base_id}, expected ${EXPECTED_BASE_ID}" >&2
  echo "refusing to derive r10 from an unverified r9" >&2
  exit 1
fi

cat >"${DOCKERFILE}" <<'EOF'
ARG BASE_IMAGE=jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG TRANSFORMERS_PIN=5.11.0
ARG IMAGE_GENERATION=r10

LABEL spark.image_generation="${IMAGE_GENERATION}" \
      spark.transformers_pin="${TRANSFORMERS_PIN}" \
      spark.base_image_id="${BASE_IMAGE_ID}"

RUN python3 -c 'import transformers; print("TRANSFORMERS_BEFORE_PIN", transformers.__version__)' \
 && python3 -m pip install --no-cache-dir "transformers==${TRANSFORMERS_PIN}"

RUN python3 - <<'PY'
import transformers
print("TRANSFORMERS_BAKED", transformers.__version__)
assert transformers.__version__ == "@TRANSFORMERS_PIN@", transformers.__version__
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
except Exception:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as M
assert "gemma4_unified" in M, "gemma4_unified missing from transformers config mapping"
print("GEMMA4_UNIFIED_CONFIG_MAPPING present")
from pathlib import Path
import vllm
import flashinfer
print("vllm", getattr(vllm, "__version__", None), vllm.__file__)
print("flashinfer", getattr(flashinfer, "__version__", None), flashinfer.__file__)
pkg = Path(vllm.__file__).resolve().parent
for rel in [
    "_C.abi3.so",
    "_C_stable_libtorch.abi3.so",
    "_moe_C.abi3.so",
    "vllm_flash_attn/_vllm_fa2_C.abi3.so",
]:
    path = pkg / rel
    print(rel, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else 0)
    assert path.exists(), f"missing {rel}"
PY

# FlashInfer module-cache hygiene, same dir set as the r8/r9 precacheclean
# step, plus the pip cache from the pin layer.
RUN rm -rf /root/.cache/flashinfer /root/.cache/flashinfer-aiter \
      /tmp/flashinfer /tmp/flashinfer_modules /root/.cache/pip \
 && for d in /root/.cache/flashinfer /root/.cache/flashinfer-aiter /tmp/flashinfer /tmp/flashinfer_modules; do \
      echo "CACHE_AUDIT ${d} exists=$([ -e ${d} ] && echo yes || echo no)"; \
    done
EOF

# The Dockerfile heredoc is single-quoted so the build-time python assert can
# not see shell expansion; substitute the pin literal explicitly.
sed -i "s/@TRANSFORMERS_PIN@/${TRANSFORMERS_PIN}/" "${DOCKERFILE}"

{
  echo "# vLLM Gemma4 Rebuilt-C Image Build ${IMAGE_GENERATION}"
  echo
  echo "- image: \`${IMAGE_TAG}\`"
  echo "- base image: \`${BASE_IMAGE}\` (id \`${actual_base_id}\`, id-pinned)"
  echo "- transformers pin: \`${TRANSFORMERS_PIN}\`"
  echo "- image generation: \`${IMAGE_GENERATION}\`"
  echo "- log: \`${LOG_PATH}\`"
  echo "- started JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: running"
} >"${SUMMARY_PATH}"

if docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg BASE_IMAGE_ID="${actual_base_id}" \
  --build-arg TRANSFORMERS_PIN="${TRANSFORMERS_PIN}" \
  --build-arg IMAGE_GENERATION="${IMAGE_GENERATION}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" "${BUILD_CONTEXT}" 2>&1 | tee "${LOG_PATH}"; then
  IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")
  IMAGE_SIZE=$(docker image inspect --format '{{.Size}}' "${IMAGE_TAG}")
  STATUS=built
else
  STATUS=failed
  IMAGE_ID=none
  IMAGE_SIZE=0
fi

{
  echo "# vLLM Gemma4 Rebuilt-C Image Build ${IMAGE_GENERATION}"
  echo
  echo "- image: \`${IMAGE_TAG}\`"
  echo "- image id: \`${IMAGE_ID}\`"
  echo "- image size bytes: \`${IMAGE_SIZE}\`"
  echo "- base image: \`${BASE_IMAGE}\` (id \`${actual_base_id}\`, id-pinned)"
  echo "- vLLM ref: \`9759e3b06baa85db93e10ecc0a8afdc4199f449b\` (inherited from r9)"
  echo "- FlashInfer ref: \`76af798243d11c4910eaceaf1d62ba4227656d4a\` (inherited from r9)"
  echo "- transformers pin: \`${TRANSFORMERS_PIN}\`"
  echo "- image generation: \`${IMAGE_GENERATION}\`"
  echo "- log: \`${LOG_PATH}\`"
  echo "- finished JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: ${STATUS}"
  echo
  echo "Post-build provenance gates (GPU, run separately, banked with the"
  echo "serving rows): import probe incl. CC/SM count, cuobjdump sm_121a"
  echo "cubins on _C.abi3.so, nvfp4_linear_latch_diag.py, module-cache audit."
} >"${SUMMARY_PATH}"

echo "wrote ${LOG_PATH}"
echo "wrote ${SUMMARY_PATH}"
test "${STATUS}" = built
