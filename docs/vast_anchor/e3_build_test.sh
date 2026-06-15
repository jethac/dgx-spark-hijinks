#!/usr/bin/env bash
# Build vLLM-e3 (v0.23.0 + our patches) from source + FlashInfer-e3 (main + our patches) JIT,
# then run the 12B/31B matched green ladder to validate the epoch-3 rebase on sm_120.
set -e
export DEBIAN_FRONTEND=noninteractive
echo "=== apt + venv + torch ==="
apt-get update -q >/dev/null 2>&1
apt-get install -y -q python3.12-venv python3.12-dev build-essential git wget cmake ninja-build >/dev/null 2>&1
python3 -m venv /root/v
/root/v/bin/pip install -q -U pip setuptools wheel setuptools_scm >/dev/null 2>&1
/root/v/bin/pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q ninja transformers pyarrow numpy 2>&1 | tail -1
echo "=== clone e3 branches ==="
cd /root
git clone -q -b spark/hijinks-e3-flashinfer https://github.com/jethac/flashinfer flashinfer 2>&1 | tail -1
git -C flashinfer submodule update --init --recursive --depth 1 >/dev/null 2>&1
mkdir -p flashinfer/flashinfer/data
ln -sfn ../../csrc flashinfer/flashinfer/data/csrc; ln -sfn ../../include flashinfer/flashinfer/data/include
ln -sfn ../../3rdparty/cutlass flashinfer/flashinfer/data/cutlass; ln -sfn ../../3rdparty/cccl flashinfer/flashinfer/data/cccl; ln -sfn ../../3rdparty/spdlog flashinfer/flashinfer/data/spdlog
git clone -q -b spark/hijinks-e3-vllm https://github.com/jethac/vllm vllm-e3 2>&1 | tail -1
echo "=== build vLLM-e3 from source (sm_120a; this is the long step) ==="
cd /root/vllm-e3
export TORCH_CUDA_ARCH_LIST="12.0a" VLLM_TARGET_DEVICE=cuda MAX_JOBS=$(nproc) CCACHE_DIR=/root/.ccache
/root/v/bin/pip install -e . --no-build-isolation 2>&1 | tail -25
echo "=== import verify ==="
PYTHONPATH=/root/flashinfer /root/v/bin/python -c "import vllm,flashinfer,torch; print('OK vllm',vllm.__version__,'torch',torch.__version__)"
echo DONE_E3BUILD
