#!/usr/bin/env python3
"""Per-position NLL for single (nbt8192) vs chunked (nbt4096) nvfp4, ctx 8185, to see WHICH scored
positions inflate (uniform vs concentrated near the chunk boundary)."""
import os, json, math, torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
MID="google/gemma-4-12b-it"; CTX=8185; PSTART=4096
tok=AutoTokenizer.from_pretrained(MID, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
text=open("/root/wikitext_8k.txt",encoding="utf-8").read()
ids=tok.encode(text, add_special_tokens=False)[:CTX]
def run(nbt):
    llm=LLM(model=MID, kv_cache_dtype="nvfp4", max_model_len=8192, max_num_batched_tokens=nbt,
            gpu_memory_utilization=0.5, trust_remote_code=True, enforce_eager=True,
            enable_prefix_caching=True)
    o=llm.generate([{"prompt_token_ids":ids}], SamplingParams(max_tokens=1,temperature=0.0,prompt_logprobs=1))[0]
    pls=o.prompt_logprobs
    nll=[]
    for i in range(PSTART+1, len(ids)):
        e=pls[i]
        lp=None
        if e:
            for k,v in e.items():
                if k==ids[i] or (isinstance(k,str) and k.lstrip("token_id:")==str(ids[i])):
                    lp=getattr(v,"logprob",v if isinstance(v,(int,float)) else None)
        nll.append(-lp if lp is not None else float("nan"))
    del llm; torch.cuda.empty_cache()
    return nll
s=run(8192); c=run(4096)
import statistics
print(f"single mean={statistics.fmean([x for x in s if x==x]):.4f}  chunked mean={statistics.fmean([x for x in c if x==x]):.4f}")
# per-position-bucket delta (position relative to PSTART)
N=len(s); bk=[(0,256),(256,1024),(1024,2048),(2048,3072),(3072,N)]
print("relpos-bin   single   chunked   delta")
for lo,hi in bk:
    ss=[s[i] for i in range(lo,min(hi,N)) if s[i]==s[i]]; cc=[c[i] for i in range(lo,min(hi,N)) if c[i]==c[i]]
    if ss and cc: print(f"[{lo:5d},{hi:5d})  {statistics.fmean(ss):.4f}  {statistics.fmean(cc):.4f}  {statistics.fmean(ss)-statistics.fmean(cc):+.4f}")
json.dump({"single":s,"chunked":c}, open("/root/per_pos.json","w"))
print("DONE_PERPOS")
