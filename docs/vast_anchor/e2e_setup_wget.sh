set -e
export DEBIAN_FRONTEND=noninteractive
WHEEL_URL="https://github.com/jethac/vllm/releases/download/sm120a-wheels-6adc00f70/vllm-0.1.dev1%2Bg6adc00f70.sm120a-cp312-cp312-linux_x86_64.whl"
WHEEL=/root/vllm-0.1.dev1+g6adc00f70.sm120a-cp312-cp312-linux_x86_64.whl
echo "=== apt ==="; apt-get update -q >/dev/null 2>&1; apt-get install -y -q python3.12-venv python3.12-dev build-essential git wget >/dev/null 2>&1
echo "=== venv+torch ==="; python3 -m venv /root/v; /root/v/bin/pip install -q -U pip >/dev/null 2>&1
/root/v/bin/pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q ninja transformers pyarrow 2>&1 | tail -1
echo "=== wget wheel (box download, fast) ==="; wget -q -O "$WHEEL" "$WHEEL_URL"; ls -la "$WHEEL" | awk '{print "wheel size:",$5}'
echo "=== install vllm wheel ==="; /root/v/bin/pip install -q "$WHEEL" 2>&1 | tail -2
echo "=== flashinfer ==="; cd /root
git clone -q https://github.com/jethac/flashinfer 2>&1 | tail -1
git -C flashinfer fetch -q origin 7d5d477b7725943c8f1242490d38e88aa3d99e19 2>&1 | tail -1 || git -C flashinfer fetch -q --all 2>&1 | tail -1
git -C flashinfer checkout -q 7d5d477b7725943c8f1242490d38e88aa3d99e19
git -C flashinfer submodule update --init --recursive --depth 1 >/dev/null 2>&1
mkdir -p flashinfer/flashinfer/data
ln -sfn ../../csrc flashinfer/flashinfer/data/csrc; ln -sfn ../../include flashinfer/flashinfer/data/include
ln -sfn ../../3rdparty/cutlass flashinfer/flashinfer/data/cutlass; ln -sfn ../../3rdparty/cccl flashinfer/flashinfer/data/cccl; ln -sfn ../../3rdparty/spdlog flashinfer/flashinfer/data/spdlog
echo "=== verify ==="
PYTHONPATH=/root/flashinfer /root/v/bin/python -c "import vllm,flashinfer,torch; print('OK vllm',vllm.__version__,'torch',torch.__version__)"
echo DONE_SETUP
