#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_vllm_incontainer_target_audit.sh IMAGE RUN_ID

Runs vLLM/FlashInfer CUDA target audits inside a container image instead of the
host Python environment. This is intended for Spark-class GB10 image checks
where host-side imports cannot see container-local vLLM or FlashInfer packages.

Environment:
  RESULTS_DIR=results            artifact output directory
  DOCKER_PLATFORM=linux/arm64    platform for docker image inspect metadata only
  PACKAGES=vllm,flashinfer       comma-separated Python packages to audit
  MAX_FILES=120                  max .so files per package root
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

IMAGE=$1
RUN_ID=$2
RESULTS_DIR=${RESULTS_DIR:-results}
PACKAGES=${PACKAGES:-vllm,flashinfer}
MAX_FILES=${MAX_FILES:-120}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${RESULTS_DIR}"

docker image inspect "${IMAGE}" > "${RESULTS_DIR}/${RUN_ID}_image_inspect.json"

docker run --rm --gpus all --ipc=host --network=host \
  -e RUN_ID="${RUN_ID}" \
  -e RESULTS_DIR="/workspace/dgx-spark-hijinks/${RESULTS_DIR}" \
  -e PACKAGES="${PACKAGES}" \
  -e MAX_FILES="${MAX_FILES}" \
  -e HOST_IMAGE="${IMAGE}" \
  -v "${REPO_ROOT}:/workspace/dgx-spark-hijinks" \
  -w /workspace/dgx-spark-hijinks \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail

python3 - <<'"'"'PY'"'"' >"${RESULTS_DIR}/${RUN_ID}_incontainer_versions.json"
import importlib
import json
import os

def package_record(name):
    try:
        mod = importlib.import_module(name)
        return {
            "file": getattr(mod, "__file__", None),
            "version": getattr(mod, "__version__", None),
        }
    except Exception as exc:
        return {"error": repr(exc)}

packages = {
    name: package_record(name)
    for name in [item for item in os.environ["PACKAGES"].split(",") if item]
}

record = {
    "schema": "container-runtime-versions/v1",
    "packages": packages,
}

try:
    import torch

    record.update(
        {
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count
            if torch.cuda.is_available()
            else None,
            "total_memory": torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None,
        }
    )
except Exception as exc:
    record["torch_error"] = repr(exc)

print(json.dumps(record, indent=2, sort_keys=True))
PY

AUDIT_ARGS=()
IFS="," read -r -a PKG_ARRAY <<< "${PACKAGES}"
for package in "${PKG_ARRAY[@]}"; do
  if [[ -n "${package}" ]]; then
    AUDIT_ARGS+=(--package "${package}")
  fi
done

python3 scripts/cuda_so_audit.py \
  "${AUDIT_ARGS[@]}" \
  --max-files "${MAX_FILES}" \
  --output "${RESULTS_DIR}/${RUN_ID}_incontainer_cuda_so_audit.json"

ARTIFACT_PATH_ARGS=()
for cache_path in /root/.cache /tmp /var/tmp; do
  if [[ -e "${cache_path}" ]]; then
    ARTIFACT_PATH_ARGS+=(--path "${cache_path}")
  fi
done

python3 scripts/cuda_artifact_arch_audit.py \
  "${AUDIT_ARGS[@]}" \
  "${ARTIFACT_PATH_ARGS[@]}" \
  --max-files "${MAX_FILES}" \
  --output "${RESULTS_DIR}/${RUN_ID}_incontainer_cuda_artifact_arch_audit.json"

python3 - <<'"'"'PY'"'"' >"${RESULTS_DIR}/${RUN_ID}_incontainer_target_audit.md"
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
results_dir = Path(os.environ["RESULTS_DIR"])
versions_path = results_dir / f"{run_id}_incontainer_versions.json"
cuda_so_path = results_dir / f"{run_id}_incontainer_cuda_so_audit.json"
artifact_path = results_dir / f"{run_id}_incontainer_cuda_artifact_arch_audit.json"
versions = json.loads(versions_path.read_text())
cuda_so = json.loads(cuda_so_path.read_text())
artifact = json.loads(artifact_path.read_text())
summary = cuda_so.get("summary", {})
artifact_summary = artifact.get("summary", {})
arch_counts = summary.get("architecture_counts", {})
artifact_arch_counts = artifact_summary.get("architecture_counts", {})
host_image = os.environ.get("HOST_IMAGE", "unknown")
device_name = versions.get("device_name")
device_capability = versions.get("device_capability")
multi_processor_count = versions.get("multi_processor_count")
torch_version = versions.get("torch")
torch_cuda = versions.get("torch_cuda")
torch_arch_list = versions.get("arch_list")
packages = versions.get("packages")
object_count = summary.get("object_count")
objects_with_sm_120 = summary.get("objects_with_sm_120")
objects_with_sm_121 = summary.get("objects_with_sm_121")
artifacts_with_sm_121 = artifact_summary.get("artifacts_with_sm_121")
artifacts_with_sm_121a = artifact_summary.get("artifacts_with_sm_121a")
artifacts_with_compute_121 = artifact_summary.get("artifacts_with_compute_121")
artifacts_with_compute_121a = artifact_summary.get("artifacts_with_compute_121a")

def bullet(value):
    return "none" if not value else ", ".join(f"{k}={v}" for k, v in sorted(value.items()))

lines = [
    f"# In-Container Target Audit: {run_id}",
    "",
    f"Image: `{host_image}`",
    "",
    "Artifacts:",
    "",
    f"- runtime versions: `{versions_path}`",
    f"- CUDA object audit: `{cuda_so_path}`",
    f"- CUDA artifact/JIT-cache audit: `{artifact_path}`",
    "",
    "Findings:",
    "",
    f"- Device: `{device_name}`, capability `{device_capability}`, SMs `{multi_processor_count}`.",
    f"- Torch: `{torch_version}`, CUDA `{torch_cuda}`.",
    f"- Torch arch list: `{torch_arch_list}`.",
    f"- Package roots: `{packages}`.",
    f"- Inspected CUDA objects: `{object_count}`.",
    f"- Architecture counts: {bullet(arch_counts)}.",
    f"- Objects with `sm_120`: `{objects_with_sm_120}`.",
    f"- Objects with `sm_121`: `{objects_with_sm_121}`.",
    f"- Artifact/JIT architecture counts: {bullet(artifact_arch_counts)}.",
    f"- Artifacts with `sm_121`: `{artifacts_with_sm_121}`.",
    f"- Artifacts with `sm_121a`: `{artifacts_with_sm_121a}`.",
    f"- Artifacts with `compute_121`: `{artifacts_with_compute_121}`.",
    f"- Artifacts with `compute_121a`: `{artifacts_with_compute_121a}`.",
    "",
]

has_native = bool(
    summary.get("objects_with_sm_121")
    or artifact_summary.get("artifacts_with_sm_121")
    or artifact_summary.get("artifacts_with_sm_121a")
    or artifact_summary.get("artifacts_with_compute_121")
    or artifact_summary.get("artifacts_with_compute_121a")
)
if has_native:
    lines.append("Conclusion: inspected CUDA objects include explicit `sm_121`/`sm_121a` target evidence.")
else:
    lines.append(
        "Conclusion: inspected CUDA objects do not include explicit `sm_121` target evidence. "
        "Treat this as functional compatibility evidence unless separate JIT logs or profiler traces prove native Spark targets on the serving critical path."
    )

print("\n".join(lines))
PY
'

echo "wrote ${RESULTS_DIR}/${RUN_ID}_incontainer_versions.json"
echo "wrote ${RESULTS_DIR}/${RUN_ID}_incontainer_cuda_so_audit.json"
echo "wrote ${RESULTS_DIR}/${RUN_ID}_incontainer_cuda_artifact_arch_audit.json"
echo "wrote ${RESULTS_DIR}/${RUN_ID}_incontainer_target_audit.md"
