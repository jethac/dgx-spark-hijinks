#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-nvcr.io/nvidia/sglang:26.05-py3}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-}
PREPARE_RUST_IMAGE=${PREPARE_RUST_IMAGE:-1}
RUN_ID=${RUN_ID:-sglang_kernel_source_build_$(date -u +%Y%m%dT%H%M%SZ)}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results}
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}

mkdir -p "${RESULTS_DIR}"

if [[ -z "${RUNTIME_IMAGE}" ]]; then
  RUNTIME_IMAGE="${IMAGE}"
fi

if [[ "${PREPARE_RUST_IMAGE}" == "1" ]]; then
  image_tag=$(
    printf '%s' "${RUN_ID}" |
      tr '[:upper:]' '[:lower:]' |
      tr -c 'a-z0-9_.-' '-'
  )
  RUNTIME_IMAGE="sglang-kernel-probe-${image_tag}-rust"
  docker build -t "${RUNTIME_IMAGE}" - <<EOF
FROM ${IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:\${PATH}"
EOF
fi

container="${RUN_ID}"
log="results/${RUN_ID}.log"
summary="results/${RUN_ID}_summary.json"

docker rm -f "${container}" >/dev/null 2>&1 || true

docker run --rm --name "${container}" --gpus all --ipc=host --network=host \
  -v "${REPO_ROOT}:/work" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -w /work \
  "${RUNTIME_IMAGE}" \
  bash -lc '
set -euxo pipefail
git config --global --add safe.directory /work
git config --global --add safe.directory /work/third_party/flashinfer
git config --global --add safe.directory /work/third_party/sglang

python3 - <<PY
import json, pathlib, subprocess, torch
out = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
pathlib.Path("/work/'"${summary}"'").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY

python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache sglang-kernel || true
rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer \
       /usr/local/lib/python3.12/dist-packages/flashinfer_python-*.dist-info \
       /usr/local/lib/python3.12/dist-packages/flashinfer_cubin* \
       /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache* \
       /usr/local/lib/python3.12/dist-packages/sgl_kernel \
       /usr/local/lib/python3.12/dist-packages/sglang_kernel-*.dist-info \
       /root/.cache/flashinfer || true

python3 -m pip install --upgrade --no-deps "nvidia-cutlass-dsl[cu13]>=4.5.0" scikit-build-core ninja cmake wheel
python3 -m pip install --no-deps --no-build-isolation -e /work/third_party/flashinfer

cd /work/third_party/sglang/sgl-kernel
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" \
MAX_JOBS="${MAX_JOBS:-2}" \
CMAKE_ARGS="${CMAKE_ARGS:--DSGL_KERNEL_COMPILE_THREADS=1 -DENABLE_BELOW_SM90=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSGL_KERNEL_BUILD_SM90=OFF -DSGL_KERNEL_BUILD_SM100=ON -DSGL_KERNEL_ENABLE_FA3=OFF -DSGL_KERNEL_ENABLE_FLASHMLA=OFF -DSGL_KERNEL_ENABLE_SPATIAL=OFF}" \
python3 -m pip install --no-deps --no-build-isolation -v .

cd /work
python3 - <<PY
import glob, importlib.metadata as md, json, pathlib, sgl_kernel, torch
root = pathlib.Path(sgl_kernel.__file__).parent
out = json.loads(pathlib.Path("/work/'"${summary}"'").read_text())
out.update(
    {
        "sglang_kernel": md.version("sglang-kernel"),
        "sgl_kernel_file": sgl_kernel.__file__,
        "common_ops_sm90": glob.glob(str(root / "sm90" / "common_ops.*")),
        "common_ops_sm100": glob.glob(str(root / "sm100" / "common_ops.*")),
        "common_ops_root": glob.glob(str(root / "common_ops.*")),
        "loaded_common_ops": getattr(getattr(sgl_kernel, "common_ops", None), "__file__", None),
    }
)
pathlib.Path("/work/'"${summary}"'").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
PY
' 2>&1 | tee "${RESULTS_DIR}/${RUN_ID}.log"

cat "${RESULTS_DIR}/${RUN_ID}_summary.json"
