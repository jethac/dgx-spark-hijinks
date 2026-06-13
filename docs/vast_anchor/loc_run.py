import torch, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tk=os.environ["HF_TOKEN"]; MID="google/gemma-4-12B"
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.])
def e2m1(mag):
    lv=E2M1.to(mag.device); mids=(lv[1:]+lv[:-1])/2; return lv[torch.bucketize(mag.clamp(max=6.),mids)]
def qdq(x,block):
    D=x.shape[-1]; lead=x.shape[:-1]
    if block=="tensor":
        amax=x.abs().amax(); s=(amax/6.).clamp(min=1e-8).to(torch.float8_e4m3fn).float(); xs=x.float()/s
        return (e2m1(xs.abs())*xs.sign()*s).to(x.dtype)
    if D%block: return x
    xb=x.reshape(*lead,D//block,block).float(); amax=xb.abs().amax(-1,keepdim=True)
    s=(amax/6.).clamp(min=1e-8).to(torch.float8_e4m3fn).float(); xs=xb/s
    return (e2m1(xs.abs())*xs.sign()*s).reshape(*lead,D).to(x.dtype)
_o=torch.nn.functional.scaled_dot_product_attention; CFG=[None]
def patched(q,k,v,*a,**kw):
    c=CFG[0]
    if c:
        blk,kk,vv=c
        if kk:k=qdq(k,blk)
        if vv:v=qdq(v,blk)
    return _o(q,k,v,*a,**kw)
torch.nn.functional.scaled_dot_product_attention=patched
tok=AutoTokenizer.from_pretrained(MID,token=tk)
m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map="cuda",token=tk,attn_implementation="sdpa")
fp=hf_hub_download("Salesforce/wikitext","wikitext-2-raw-v1/test-00000-of-00001.parquet",repo_type="dataset",token=tk)
text="\n".join(x for x in pq.read_table(fp).column("text").to_pylist() if x and x.strip())
ids=tok(text)["input_ids"][:4096]; inp=torch.tensor([ids],device="cuda"); tgt=torch.tensor(ids[1:],device="cuda")
def nll(cfg):
    CFG[0]=cfg
    with torch.no_grad(): lg=m(inp).logits[0]
    tot=0.;n=0
    for i in range(0,len(ids)-1,256):
        ch=lg[i:i+256].float(); lp=torch.log_softmax(ch,-1)
        t=tgt[i:i+256]; tot+= -lp[range(ch.shape[0]),t].sum().item(); n+=ch.shape[0]
    return tot/n
base=nll(None); print(f"bf16 baseline NLL={base:.4f}")
for label,cfg in [("block16 K+V",(16,1,1)),("block32 K+V",(32,1,1)),("block64 K+V",(64,1,1)),("block128 K+V",(128,1,1)),("block256 K+V",(256,1,1)),("per-tensor K+V",("tensor",1,1)),("block16 K-only",(16,1,0)),("block16 V-only",(16,0,1))]:
    print(f"  {label:20s} DELTA={nll(cfg)-base:+.4f}")
