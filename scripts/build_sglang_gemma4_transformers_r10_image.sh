#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_sglang_gemma4_transformers_r10_image.sh [IMAGE_TAG]

Derive a SGLang Gemma 4 runtime image from an existing source-stack image and
bake the Transformers version that recognizes `gemma4_unified` (Gemma 4 12B).

This is intentionally a light image layer: it does not rebuild SGLang,
sgl-kernel, or FlashInfer. Use it after a source-stack image has already been
compiled and verified.

Environment:
  BASE_IMAGE=sglang-source-stack-dgemma-024-0705924c-f99323bd:latest
  TRANSFORMERS_PIN=5.11.0
  RESULTS_DIR=/home/jethac/dgx-spark-hijinks/results
  RUN_ID=sglang_gemma4_transformers_r10_YYYYMMDDTHHMMJST
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

IMAGE_TAG=${1:-sglang-source-stack-dgemma-024-0705924c-f99323bd-tf511-r10:latest}
BASE_IMAGE=${BASE_IMAGE:-sglang-source-stack-dgemma-024-0705924c-f99323bd:latest}
TRANSFORMERS_PIN=${TRANSFORMERS_PIN:-5.11.0}
RUN_ID=${RUN_ID:-sglang_gemma4_transformers_r10_$(TZ=Asia/Tokyo date +%Y%m%dT%H%MJST)}
RESULTS_DIR=${RESULTS_DIR:-/home/jethac/dgx-spark-hijinks/results}
LOG_PATH=${RESULTS_DIR}/${RUN_ID}.log
SUMMARY_PATH=${RESULTS_DIR}/${RUN_ID}_summary.md

mkdir -p "${RESULTS_DIR}"

base_id=$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE}")
dockerfile=$(mktemp)
context_dir=$(mktemp -d)
trap 'rm -f "${dockerfile}"; rm -rf "${context_dir}"' EXIT

cat >"${dockerfile}" <<'EOF'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG TRANSFORMERS_PIN=5.11.0

LABEL spark.image_generation="sglang-gemma4-transformers-r10" \
      spark.transformers_pin="${TRANSFORMERS_PIN}" \
      spark.base_image_id="${BASE_IMAGE_ID}"

RUN python3 - <<'PY'
import importlib.metadata as md
import flashinfer
import sgl_kernel
import sglang
import transformers

print("BEFORE_TRANSFORMERS", transformers.__version__)
print("BEFORE_SGLANG", md.version("sglang"), sglang.__file__)
print("BEFORE_FLASHINFER", getattr(flashinfer, "__version__", None), flashinfer.__file__)
print("BEFORE_SGL_KERNEL", md.version("sglang-kernel"), sgl_kernel.__file__)
PY

RUN python3 -m pip install --no-cache-dir "transformers==${TRANSFORMERS_PIN}"

RUN python3 - <<'PY'
import importlib.metadata as md
import flashinfer
import sgl_kernel
import sglang
import transformers

print("TRANSFORMERS_BAKED", transformers.__version__)
assert transformers.__version__ == "@TRANSFORMERS_PIN@", transformers.__version__
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as mapping
except Exception:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as mapping
assert "gemma4_unified" in mapping, "gemma4_unified missing from transformers config mapping"
print("GEMMA4_UNIFIED_CONFIG_MAPPING present")
print("SGLANG", md.version("sglang"), sglang.__file__)
print("FLASHINFER", getattr(flashinfer, "__version__", None), flashinfer.__file__)
print("SGLANG_KERNEL", md.version("sglang-kernel"), sgl_kernel.__file__)
PY

RUN rm -rf /root/.cache/flashinfer /root/.cache/flashinfer-aiter \
      /tmp/flashinfer /tmp/flashinfer_modules /root/.cache/pip \
 && for d in /root/.cache/flashinfer /root/.cache/flashinfer-aiter /tmp/flashinfer /tmp/flashinfer_modules; do \
      echo "CACHE_AUDIT ${d} exists=$([ -e ${d} ] && echo yes || echo no)"; \
    done
EOF

sed -i "s/@TRANSFORMERS_PIN@/${TRANSFORMERS_PIN}/" "${dockerfile}"

{
  echo "# SGLang Gemma 4 Transformers r10 Image Build"
  echo
  echo "- image: \`${IMAGE_TAG}\`"
  echo "- base image: \`${BASE_IMAGE}\`"
  echo "- base image id: \`${base_id}\`"
  echo "- transformers pin: \`${TRANSFORMERS_PIN}\`"
  echo "- log: \`${LOG_PATH}\`"
  echo "- started JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: running"
} >"${SUMMARY_PATH}"

if docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg BASE_IMAGE_ID="${base_id}" \
  --build-arg TRANSFORMERS_PIN="${TRANSFORMERS_PIN}" \
  -t "${IMAGE_TAG}" \
  -f "${dockerfile}" "${context_dir}" 2>&1 | tee "${LOG_PATH}"; then
  image_id=$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")
  image_size=$(docker image inspect --format '{{.Size}}' "${IMAGE_TAG}")
  status=built
else
  image_id=none
  image_size=0
  status=failed
fi

{
  echo "# SGLang Gemma 4 Transformers r10 Image Build"
  echo
  echo "- image: \`${IMAGE_TAG}\`"
  echo "- image id: \`${image_id}\`"
  echo "- image size bytes: \`${image_size}\`"
  echo "- base image: \`${BASE_IMAGE}\`"
  echo "- base image id: \`${base_id}\`"
  echo "- transformers pin: \`${TRANSFORMERS_PIN}\`"
  echo "- log: \`${LOG_PATH}\`"
  echo "- finished JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: ${status}"
} >"${SUMMARY_PATH}"

[[ "${status}" == built ]]
