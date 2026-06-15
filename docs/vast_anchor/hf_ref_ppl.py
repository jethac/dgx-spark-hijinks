#!/usr/bin/env python3
"""HF transformers eager bf16 ground-truth suffix-NLL, matching vllm_matched_kv_anchor.py exactly:
token_ids = tokenizer.encode(text)[:ctx]; score positions prefix_len+1 .. ctx-1 (the same 4088-token
window the anchor scores). This is the dtype/kernel-independent truth to adjudicate a suspicious vLLM row.
Usage: hf_ref_ppl.py <model> <corpus> [ctx=8185] [prefix=4096]"""
import sys, json, math, torch
from transformers import AutoTokenizer

model, corpus = sys.argv[1], sys.argv[2]
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 8185
prefix = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
text = open(corpus, encoding="utf-8", errors="replace").read()
ids = tok.encode(text, add_special_tokens=False)[:ctx]

# load full model (Gemma4 may be a conditional-generation/multimodal class); text-only forward
try:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda",
                                             trust_remote_code=True, attn_implementation="eager")
except Exception as e:
    sys.stderr.write(f"AutoModelForCausalLM failed ({e}); trying ImageTextToText\n")
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda",
                                                    trust_remote_code=True, attn_implementation="eager")
m.eval()
inp = torch.tensor([ids], device="cuda")
with torch.no_grad():
    out = m(input_ids=inp)
logits = out.logits[0].float()                       # [seq, vocab]
logp = torch.log_softmax(logits, dim=-1)
nlls = [-logp[i - 1, ids[i]].item() for i in range(prefix + 1, ctx)]  # predict ids[i] from prefix [0:i]
mean = sum(nlls) / len(nlls)
print("HFREF " + json.dumps({"model": model, "mean_nll_nats": round(mean, 4),
                             "ppl": round(math.exp(mean), 4), "n": len(nlls)}))
