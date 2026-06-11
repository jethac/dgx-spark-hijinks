#!/bin/bash
# Clone FlashInfer (source-tree mode, dispatcher-fix rev) + deps. Run as user.
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
source ~/sm120env/bin/activate

if [ ! -d ~/flashinfer/.git ]; then
    git clone --recursive https://github.com/jethac/flashinfer -b spark/hijinks-022-fa2-d512 ~/flashinfer
fi
cd ~/flashinfer
git checkout 76af7982
git submodule update --init --recursive
echo "FLASHINFER_REV=$(git rev-parse HEAD)"

pip install --quiet ninja jinja2 packaging apache-tvm-ffi filelock requests einops nvidia-ml-py
echo "DEPS_DONE"

# Sanity: import flashinfer from source tree
PYTHONPATH=~/flashinfer python -c "import flashinfer; print('FLASHINFER_IMPORT_OK', flashinfer.__file__)"
