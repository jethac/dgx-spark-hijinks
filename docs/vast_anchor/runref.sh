#!/usr/bin/env bash
export HF_TOKEN="$1"
export PYTHONPATH=/root/flashinfer
export PATH=/root/v/bin:$PATH
cd /root
[ -f /root/wikitext_8k.txt ] || python corpus_fetch.py
REFMID=google/gemma-4-12b-it REFCTX=8185 REFPSTART=4096 python refsim_longctx.py
