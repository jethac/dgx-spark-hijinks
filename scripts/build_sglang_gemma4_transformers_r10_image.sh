#!/usr/bin/env bash
set -euo pipefail

# Historical local derivative builder only. The current SGLang Spark carrier is
# built by .github/workflows/hijinks-sglang-gemma4-source-stack-image.yml and
# already bakes Transformers 5.11 plus the mm-prefix SGLang changes
# (`epoch2-sglang-mm-prefix-f920e2d-arm64`,
# sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714).
# Refuse by default so we do not accidentally derive a new image from the older
# 0d5e160 base and lose the mm-prefix fix.
if [[ "${ALLOW_SUPERSEDED_SGLANG_R10_BUILDER:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
build_sglang_gemma4_transformers_r10_image.sh is superseded and disabled.

Use the GitHub/Ubicloud source-stack workflow:
  .github/workflows/hijinks-sglang-gemma4-source-stack-image.yml

Only replay this local derivative builder with
ALLOW_SUPERSEDED_SGLANG_R10_BUILDER=1 when intentionally reproducing the old
r10 Transformers-overlay image from the pre-mm-prefix source stack.
EOF
  exit 2
fi

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_sglang_gemma4_transformers_r10_image.sh [IMAGE_TAG]

Derive a SGLang Gemma 4 runtime image from an existing source-stack image and
bake the Transformers version that recognizes `gemma4_unified` (Gemma 4 12B).

This is intentionally a light image layer: it does not rebuild SGLang,
sgl-kernel, or FlashInfer. Use it after a source-stack image has already been
compiled and verified.

Environment:
  BASE_IMAGE=ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94
  TRANSFORMERS_PIN=5.11.0
  RESULTS_DIR=/home/jethac/dgx-spark-hijinks/results
  RUN_ID=sglang_gemma4_transformers_r10_YYYYMMDDTHHMMJST
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

IMAGE_TAG=${1:-sglang-gemma4-source-stack-u22-torch211-tf511-r10:latest}
BASE_IMAGE=${BASE_IMAGE:-ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94}
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
ARG BASE_IMAGE=ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG TRANSFORMERS_PIN=5.11.0

LABEL spark.image_generation="sglang-gemma4-transformers-r10" \
      spark.transformers_pin="${TRANSFORMERS_PIN}" \
      spark.base_image_id="${BASE_IMAGE_ID}"

RUN python3 -c 'import transformers; print("BEFORE_TRANSFORMERS", transformers.__version__)'

RUN python3 -m pip install --no-cache-dir "transformers==${TRANSFORMERS_PIN}"

RUN python3 - <<'PY'
import transformers

print("TRANSFORMERS_BAKED", transformers.__version__)
assert transformers.__version__ == "@TRANSFORMERS_PIN@", transformers.__version__
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as mapping
except Exception:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as mapping
assert "gemma4_unified" in mapping, "gemma4_unified missing from transformers config mapping"
print("GEMMA4_UNIFIED_CONFIG_MAPPING present")
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

VERIFY_STATUS=not_run
VERIFY_LOG=${RESULTS_DIR}/${RUN_ID}_runtime_verify.log
if [[ "${status}" == built ]]; then
  if docker run --rm \
    -v "$(pwd):/work" \
    -w /work \
    -e PYTHONPATH=/work/third_party/sglang/python:/work/third_party/flashinfer \
    "${IMAGE_TAG}" \
    bash -lc 'python3 - <<'"'"'PY'"'"'
import importlib.metadata as md

import flashinfer
import sgl_kernel
import sglang
import transformers

print("RUNTIME_TRANSFORMERS", transformers.__version__)
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as mapping
except Exception:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING as mapping
assert "gemma4_unified" in mapping, "gemma4_unified missing from transformers config mapping"
print("RUNTIME_GEMMA4_UNIFIED_CONFIG_MAPPING present")
print("RUNTIME_SGLANG", md.version("sglang"), sglang.__file__)
print("RUNTIME_FLASHINFER", getattr(flashinfer, "__version__", None), flashinfer.__file__)
print("RUNTIME_SGLANG_KERNEL", md.version("sglang-kernel"), sgl_kernel.__file__)
print("RUNTIME_SGL_KERNEL_FILE", sgl_kernel.__file__)
PY' >"${VERIFY_LOG}" 2>&1; then
    VERIFY_STATUS=passed
  else
    VERIFY_STATUS=failed
    status=failed
  fi
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
  echo "- runtime verify: \`${VERIFY_STATUS}\`"
  echo "- runtime verify log: \`${VERIFY_LOG}\`"
  echo "- finished JST: \`$(TZ=Asia/Tokyo date --iso-8601=seconds)\`"
  echo
  echo "Status: ${status}"
} >"${SUMMARY_PATH}"

[[ "${status}" == built ]]
