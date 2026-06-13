import torch, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tk=os.environ["HF_TOKEN"]; MID=os.environ.get("REFMID","google/gemma-4-12B")

E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.])
def qdq_nvfp4(x):
    if x.shape[-1]%16: return x
    lead=x.shape[:-1]; D=x.shape[-1]
    xb=x.reshape(*lead,D//16,16).float()
    amax=xb.abs().amax(-1,keepdim=True)
    scale=(amax/6.0).clamp(min=1e-8).to(torch.float8_e4m3fn).float()
    xs=xb/scale; sign=xs.sign(); mag=xs.abs().clamp(max=6.0)
    lv=E2M1.to(x.device); mids=(lv[1:]+lv[:-1])/2
    q=lv[torch.bucketize(mag,mids)]*sign
    return (q*scale).reshape(*lead,D).to(x.dtype)

_orig=torch.nn.functional.scaled_dot_product_attention
MODE=[0]  # 0=off 1=nvfp4 K+V
def patched(q,k,v,*a,**kw):
    if MODE[0]==1: k=qdq_nvfp4(k); v=qdq_nvfp4(v)
    return _orig(q,k,v,*a,**kw)
torch.nn.functional.scaled_dot_product_attention=patched

tok=AutoTokenizer.from_pretrained(MID,token=tk)
m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map="cuda",token=tk,attn_implementation="sdpa")
fp=hf_hub_download("Salesforce/wikitext","wikitext-2-raw-v1/test-00000-of-00001.parquet",repo_type="dataset",token=tk)
text="\n".join(x for x in pq.read_table(fp).column("text").to_pylist() if x and x.strip())
def nll(n,mode):
    MODE[0]=mode
    ids=tok(text)["input_ids"][:n]; inp=torch.tensor([ids],device="cuda")
    with torch.no_grad(): lg=m(inp).logits[0].float()
    lp=torch.log_softmax(lg[:-1],-1)
    return -lp[range(len(ids)-1),torch.tensor(ids[1:],device="cuda")].mean().item()
for n in [2048,4096]:
    b=nll(n,0); q=nll(n,1)
    print(f"ctx={n}: bf16 NLL={b:.4f} (ppl {math.exp(b):.2f}) | refNVFP4 NLL={q:.4f} (ppl {math.exp(q):.2f}) | DELTA={q-b:+.4f} nats/token")
