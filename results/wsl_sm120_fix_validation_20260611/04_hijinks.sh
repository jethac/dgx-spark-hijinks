#!/bin/bash
# Clone dgx-spark-hijinks scripts. Run as user.
set -euo pipefail
if [ ! -d ~/hijinks/.git ]; then
    git clone --depth 30 https://github.com/jethac/dgx-spark-hijinks -b epoch2 ~/hijinks
fi
cd ~/hijinks
echo "HIJINKS_REV=$(git rev-parse HEAD)"
git log --oneline -1
ls scripts/vllm_gemma4_mixed_kv_probes.py scripts/flashinfer_nvfp4_kv_probe.py scripts/spark_hardware.py
echo "HIJINKS_OK"
