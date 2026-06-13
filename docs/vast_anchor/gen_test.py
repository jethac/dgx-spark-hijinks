import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
TOKEN=os.environ["HF_TOKEN"]; MODEL="google/gemma-4-12B-it"
tok=AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
llm=LLM(model=MODEL, kv_cache_dtype="bfloat16", max_model_len=8192, enforce_eager=True, gpu_memory_utilization=0.9, trust_remote_code=True)
# 1) greedy generation coherence
g=llm.generate(["The capital of France is"], SamplingParams(max_tokens=8,temperature=0.0))
print("GEN:", repr(g[0].outputs[0].text))
# 2) top-1 prefill prediction sanity: what does the model think follows each token?
ids=tok("The quick brown fox jumps over the lazy dog")["input_ids"]
o=llm.generate([{"prompt_token_ids":ids}], SamplingParams(max_tokens=1,prompt_logprobs=1,temperature=0.0))
pls=o[0].prompt_logprobs
for i in range(1,len(pls)):
    d=pls[i]
    # top-1 predicted token (max logprob) vs actual
    top=max(d.items(), key=lambda kv: kv[1].logprob)
    actual=d.get(ids[i])
    print(f"pos{i} actual={tok.decode([ids[i]])!r} nll={-actual.logprob:.2f} | top1={tok.decode([top[0]])!r} lp={top[1].logprob:.2f}")
