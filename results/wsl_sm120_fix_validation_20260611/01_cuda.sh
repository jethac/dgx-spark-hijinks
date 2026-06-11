#!/bin/bash
# Install CUDA 13.0 toolkit from NVIDIA apt repo (ubuntu2404). Run as root.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if [ -d /usr/local/cuda-13.0 ] && [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
    echo "CUDA 13.0 already installed"
    /usr/local/cuda-13.0/bin/nvcc --version
    exit 0
fi

apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git build-essential wget > /tmp/base_install.log 2>&1 || { tail -20 /tmp/base_install.log; exit 1; }
echo "BASE_PKGS_DONE"

cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-0 > /tmp/cuda_install.log 2>&1 || { tail -50 /tmp/cuda_install.log; exit 1; }
echo "CUDA_INSTALL_DONE"
/usr/local/cuda-13.0/bin/nvcc --version
