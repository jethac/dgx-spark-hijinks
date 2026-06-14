#!/usr/bin/env bash
echo "=== before (cryptic) ==="
python3 /disc/repro_d512_ragged.py 2>&1 | grep -E "qo=   8" | head -1
python3 /disc/apply_reject.py
rm -rf ~/.cache/flashinfer /root/.cache/flashinfer /tmp/flashinfer* 2>/dev/null
echo "=== after (clean reject expected) ==="
python3 /disc/repro_d512_ragged.py 2>&1 | grep -E "qo=   8|qo=  64" | head -2
echo DONE_REJECT
