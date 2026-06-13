import torch, os, sys, statistics as st, librosa
from transformers import AutoProcessor, AutoModelForImageTextToText
tk=os.environ["HF_TOKEN"]; MID=sys.argv[1]; AUD=258881
proc=AutoProcessor.from_pretrained(MID,token=tk)
model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map="cuda",token=tk,attn_implementation="sdpa")
y,sr=librosa.load(librosa.ex("libri1"),sr=16000); y=y[:16000*8]
msgs=[{"role":"user","content":[{"type":"audio","audio":y},{"type":"text","text":"What is being said in this audio?"}]}]
pin=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda")
plen=pin["input_ids"].shape[1]
gen=model.generate(**pin,max_new_tokens=24,do_sample=False)
ans=proc.decode(gen[0][plen:],skip_special_tokens=True); print(MID,"ANSWER:",repr(ans))
msgs2=msgs+[{"role":"assistant","content":[{"type":"text","text":ans}]}]
fin=proc.apply_chat_template(msgs2,tokenize=True,return_dict=True,return_tensors="pt").to("cuda")
full=fin["input_ids"]; end=full.shape[1]; amask=(full[0]==AUD)
print(MID,"seq",end,"audio",int(amask.sum().item()),"text",int((~amask).sum().item()))
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.]); FP8MAX=448.0
def e2(m): lv=E2M1.to(m.device); md=(lv[1:]+lv[:-1])/2; return lv[torch.bucketize(m.clamp(max=6.),md)]
def qdq(x,g):
    D=x.shape[-1]; L=x.shape[:-1]; xb=x.reshape(*L,D//16,16).float()
    ba=xb.abs().amax(-1,keepdim=True); ta=x.float().abs().amax().clamp(min=1e-20)
    gs=((ta/6/FP8MAX)*g).clamp(min=1e-30); sf=((ba/6)/gs).to(torch.float8_e4m3fn).float()
    bs=(sf*gs).clamp(min=1e-30); return (e2(xb.abs()/bs)*xb.sign()*bs).reshape(*L,D).to(x.dtype)
_o=torch.nn.functional.scaled_dot_product_attention; MODE=[None]; S={"ak":[],"av":[],"tk":[],"tv":[]}
def patched(q,k,v,*a,**kw):
    if MODE[0]=="stats" and k.shape[2]==end:
        if amask.any(): S["ak"].append(k[:,:,amask,:].abs().amax().item()); S["av"].append(v[:,:,amask,:].abs().amax().item())
        if (~amask).any(): S["tk"].append(k[:,:,~amask,:].abs().amax().item()); S["tv"].append(v[:,:,~amask,:].abs().amax().item())
    g=MODE[0]
    if isinstance(g,float): k=qdq(k,g); v=qdq(v,g)
    return _o(q,k,v,*a,**kw)
torch.nn.functional.scaled_dot_product_attention=patched
def nll(mode):
    MODE[0]=mode
    with torch.no_grad(): out=model(**fin)
    lg=out.logits[0].float(); lp=torch.log_softmax(lg[plen-1:end-1],-1); tgt=full[0,plen:end]
    return -lp[range(len(tgt)),tgt].mean().item()
nll("stats")
ak=st.mean(S["ak"]); av=st.mean(S["av"]); tkk=st.mean(S["tk"]); tvv=st.mean(S["tv"])
print(MID,"KVAMAX audK %.3f audV %.3f txtK %.3f txtV %.3f"%(ak,av,tkk,tvv))
print(MID,"RATIO audioOverText K %.2fx V %.2fx"%(ak/tkk,av/tvv))
b=nll(None); print(MID,"ANSWERNLL bf16 %.4f | qdq_g1_calib %+.4f | qdq_g0.5_under %+.4f"%(b,nll(1.0)-b,nll(0.5)-b))
