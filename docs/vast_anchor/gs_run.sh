set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null 2>&1; apt-get install -y -q python3.12-venv >/dev/null 2>&1
python3 -m venv /root/v; /root/v/bin/pip install -q -U pip >/dev/null 2>&1
/root/v/bin/pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
/root/v/bin/pip install -q transformers accelerate pyarrow 2>&1 | tail -1
cat > /root/gs.py <<"PYEOF"
import torch, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tk=os.environ["HF_TOKEN"]; MID="google/gemma-4-12B"; FP8MAX=448.0
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.])
def e2m1(mag):
    lv=E2M1.to(mag.device); mids=(lv[1:]+lv[:-1])/2; return lv[torch.bucketize(mag.clamp(max=6.),mids)]
def qdq2(x,g):  # two-level: per-tensor global (g=1 calibrated) + per-16 fp8 block SF
    D=x.shape[-1]; lead=x.shape[:-1]; xb=x.reshape(*lead,D//16,16).float()
    bamax=xb.abs().amax(-1,keepdim=True); tamax=x.float().abs().amax().clamp(min=1e-20)
    gscale=((tamax/6.0/FP8MAX)*g).clamp(min=1e-30)
    sf=((bamax/6.0)/gscale).to(torch.float8_e4m3fn).float()   # fp8 e4m3 store (saturates/underflows)
    bscale=(sf*gscale).clamp(min=1e-30)
    return (e2m1(xb.abs()/bscale)*xb.sign()*bscale).reshape(*lead,D).to(x.dtype)
_o=torch.nn.functional.scaled_dot_product_attention; CFG=[None]
def patched(q,k,v,*a,**kw):
    c=CFG[0]
    if c:
        g,kk,vv=c
        if kk:k=qdq2(k,g)
        if vv:v=qdq2(v,g)
    return _o(q,k,v,*a,**kw)
torch.nn.functional.scaled_dot_product_attention=patched
tok=AutoTokenizer.from_pretrained(MID,token=tk)
m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map="cuda",token=tk,attn_implementation="sdpa")
fp=hf_hub_download("Salesforce/wikitext","wikitext-2-raw-v1/test-00000-of-00001.parquet",repo_type="dataset",token=tk)
text="\n".join(x for x in pq.read_table(fp).column("text").to_pylist() if x and x.strip())
ids=tok(text)["input_ids"][:4096]; inp=torch.tensor([ids],device="cuda"); tgt=torch.tensor(ids[1:],device="cuda")
def nll(cfg):
    CFG[0]=cfg
    with torch.no_grad(): lg=m(inp).logits[0][:-1]
    tot=0.;n=0
    for i in range(0,len(tgt),256):
        ch=lg[i:i+256].float(); lp=torch.log_softmax(ch,-1); t=tgt[i:i+256]
        tot+=-lp[range(ch.shape[0]),t].sum().item(); n+=ch.shape[0]
    return tot/n
base=nll(None); print(f"bf16 baseline NLL={base:.4f}")
print("--- global-scale sweep, K+V (g=1 = calibrated/optimal) ---")
for g in [1.0,2.0,4.0,8.0,16.0,0.5,0.25]:
    print(f"  g={g:6.2f} K+V: DELTA={nll((g,1,1))-base:+.4f}")
print("--- K-only vs V-only at a few g ---")
for g in [1.0,8.0,16.0]:
    print(f"  g={g:5.1f} K-only={nll((g,1,0))-base:+.4f}  V-only={nll((g,0,1))-base:+.4f}")
PYEOF
echo "=== RUN ==="; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/v/bin/python /root/gs.py
