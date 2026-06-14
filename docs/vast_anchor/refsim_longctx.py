#!/usr/bin/env python3
"""Ground-truth long-context NVFP4-KV penalty via exact HF eager SDPA (no FlashInfer kernel).

Disambiguates the serving result: vLLM single-prefill gave Δ=+0.42 @ ctx 8185 while
chunked/reuse gave +0.19. Since both read the same all-nvfp4 paged cache, one of the two
serving paths is numerically off. This reference uses pure-torch exact softmax (no online
tiling, no VO split, no kernel), nvfp4-qdq applied to K and V before attention, and scores
the SAME suffix (positions PSTART+1..ctx-1) as the serving anchor. It also reports the
penalty binned by token position so we can see how the nvfp4 cost grows with context.
"""
import torch, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

tk = os.environ["HF_TOKEN"]
MID = os.environ.get("REFMID", "google/gemma-4-12b-it")
CTX = int(os.environ.get("REFCTX", "8185"))
PSTART = int(os.environ.get("REFPSTART", "4096"))  # score positions PSTART+1..CTX-1 (matches anchor)

E2M1 = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])
def qdq_nvfp4(x):
    if x.shape[-1] % 16:
        return x
    lead = x.shape[:-1]; D = x.shape[-1]
    xb = x.reshape(*lead, D // 16, 16).float()
    amax = xb.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0).clamp(min=1e-8).to(torch.float8_e4m3fn).float()
    xs = xb / scale; sign = xs.sign(); mag = xs.abs().clamp(max=6.0)
    lv = E2M1.to(x.device); mids = (lv[1:] + lv[:-1]) / 2
    q = lv[torch.bucketize(mag, mids)] * sign
    return (q * scale).reshape(*lead, D).to(x.dtype)

_orig = torch.nn.functional.scaled_dot_product_attention
MODE = [0]  # 0=bf16 KV, 1=nvfp4 K+V
def patched(q, k, v, *a, **kw):
    if MODE[0] == 1:
        k = qdq_nvfp4(k); v = qdq_nvfp4(v)
    return _orig(q, k, v, *a, **kw)
torch.nn.functional.scaled_dot_product_attention = patched

tok = AutoTokenizer.from_pretrained(MID, token=tk)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda", token=tk,
                                         attn_implementation="sdpa")
fp = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                     repo_type="dataset", token=tk)
text = "\n".join(x for x in pq.read_table(fp).column("text").to_pylist() if x and x.strip())
ids = tok(text)["input_ids"][:CTX]
inp = torch.tensor([ids], device="cuda")

def per_pos_nll(mode):
    MODE[0] = mode
    with torch.no_grad():
        lg = m(inp).logits[0].float()
    lp = torch.log_softmax(lg[:-1], -1)
    tgt = torch.tensor(ids[1:], device="cuda")
    return -lp[range(len(ids) - 1), tgt]  # NLL per predicted position (len CTX-1)

nb = per_pos_nll(0); nq = per_pos_nll(1)
# suffix-matched mean (predicted positions PSTART+1..CTX-1 => indices PSTART..CTX-2 in 0-based nll array)
s = slice(PSTART, len(ids) - 1)
b_suf = nb[s].mean().item(); q_suf = nq[s].mean().item()
print(f"model={MID} ctx={CTX} score_suffix=[{PSTART+1}..{CTX-1}] ({len(nb[s])} tokens)")
print(f"SUFFIX  bf16={b_suf:.4f}  nvfp4={q_suf:.4f}  DELTA={q_suf-b_suf:+.4f}")
# penalty binned by token position (effective context length)
bins = [(0,512),(512,1024),(1024,2048),(2048,4096),(4096,6144),(6144,CTX-1)]
print("position-bin   bf16     nvfp4    delta")
for lo,hi in bins:
    if lo >= len(nb): continue
    hi = min(hi, len(nb))
    d = (nq[lo:hi].mean() - nb[lo:hi].mean()).item()
    print(f"[{lo:5d},{hi:5d})  {nb[lo:hi].mean():.4f}  {nq[lo:hi].mean():.4f}  {d:+.4f}")
