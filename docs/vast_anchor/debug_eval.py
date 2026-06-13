import os, math, statistics
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
TOKEN=os.environ["HF_TOKEN"]; MODEL="google/gemma-4-12B-it"
tok=AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
llm=LLM(model=MODEL, kv_cache_dtype="bfloat16", max_model_len=8192, enforce_eager=True, gpu_memory_utilization=0.9, trust_remote_code=True)
def nll_for(ids):
    o=llm.generate([{"prompt_token_ids":ids}], SamplingParams(max_tokens=1,prompt_logprobs=1,temperature=0.0))
    pls=o[0].prompt_logprobs
    nl=[]
    for i in range(len(pls)):
        if pls[i] is None: continue
        d=pls[i]
        if ids[i] in d: nl.append(-d[ids[i]].logprob)
    return nl
# sanity sentence
s="The quick brown fox jumps over the lazy dog. The capital of France is Paris."
sid=tok(s)["input_ids"]
snl=nll_for(sid)
print("SANITY sentence: ntok",len(sid),"mean_nll",round(sum(snl)/len(snl),3),"ppl",round(math.exp(sum(snl)/len(snl)),2))
print("  first tokens:",[(repr(tok.decode([sid[i]])),round(snl[i-1],2)) for i in range(1,min(8,len(sid)))])
# wikitext
fp=hf_hub_download("Salesforce/wikitext","wikitext-2-raw-v1/test-00000-of-00001.parquet",repo_type="dataset",token=TOKEN)
texts=pq.read_table(fp).column("text").to_pylist()
text="\n".join(t for t in texts if t and t.strip())
ids=tok(text)["input_ids"][:4096]
print("wikitext head decoded:",repr(tok.decode(ids[:40])))
wnl=nll_for(ids)
wnl_s=sorted(wnl)
print("WIKITEXT: n",len(wnl),"mean",round(sum(wnl)/len(wnl),3),"median",round(statistics.median(wnl),3),"min",round(min(wnl),3),"max",round(max(wnl),3))
