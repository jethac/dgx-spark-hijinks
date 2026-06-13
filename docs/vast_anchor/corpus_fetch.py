#!/usr/bin/env python3
"""Build a contiguous >8185-token wikitext corpus file for the prefix-reuse matrix."""
import os
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

TOKEN = os.environ.get("HF_TOKEN")
fp = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                     repo_type="dataset", token=TOKEN)
texts = pq.read_table(fp).column("text").to_pylist()
text = "\n".join(t for t in texts if t and t.strip())
open("/root/wikitext_8k.txt", "w", encoding="utf-8").write(text)
print("wrote /root/wikitext_8k.txt chars=", len(text))
