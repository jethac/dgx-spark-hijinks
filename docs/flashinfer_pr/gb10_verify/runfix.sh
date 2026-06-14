#!/usr/bin/env bash
echo "=== before fix (baseline reject) ==="
python3 /disc/repro_d512_ragged.py 2>&1 | grep -E "qo=" | head -2
python3 /disc/apply_fix.py
rm -rf ~/.cache/flashinfer /root/.cache/flashinfer /tmp/flashinfer* 2>/dev/null
echo "=== after fix (should RUN) ==="
python3 /disc/repro_d512_ragged.py 2>&1 | tail -10
echo DONE_FIXVERIFY
