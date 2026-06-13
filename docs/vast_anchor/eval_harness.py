# Matched bf16-vs-NVFP4 KV PPL anchor — identical token ids, only kv_cache_dtype differs.
import sys, json, math, os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

KV = sys.argv[1]               # bfloat16 | nvfp4
OUT = sys.argv[2]
MODEL = os.environ.get("ANCHOR_MODEL", "google/gemma-4-12B-it")
MAXLEN = int(os.environ.get("ANCHOR_MAXLEN", "8192"))
NTOK = int(os.environ.get("ANCHOR_NTOK", "4096"))
TOKEN = os.environ.get("HF_TOKEN")

tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
fp = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                     repo_type="dataset", token=TOKEN)
texts = pq.read_table(fp).column("text").to_pylist()
text = "\n".join(t for t in texts if t and t.strip())
ids = tok(text)["input_ids"][:NTOK]

llm = LLM(model=MODEL, kv_cache_dtype=KV, max_model_len=MAXLEN, enforce_eager=True,
          gpu_memory_utilization=0.90, trust_remote_code=True)
o = llm.generate([{"prompt_token_ids": ids}],
                 SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
pls = o[0].prompt_logprobs
nlls = [ -pls[i][ids[i]].logprob for i in range(len(pls)) if pls[i] is not None ]
mean_nll = sum(nlls) / len(nlls)
res = {"kv": KV, "model": MODEL, "n_scored": len(nlls), "ntok": NTOK,
       "mean_nll": mean_nll, "ppl": math.exp(mean_nll)}
json.dump(res, open(OUT, "w"), indent=2)
print("RESULT", json.dumps(res))
