import torch, os, requests, statistics as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
tk=os.environ["HF_TOKEN"]; MID="google/gemma-4-E4B-it"; IMG=258880
proc=AutoProcessor.from_pretrained(MID,token=tk)
model=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.bfloat16,device_map="cuda",token=tk,attn_implementation="sdpa")
img=Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg",stream=True,timeout=60).raw).convert("RGB")
msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Describe this image in one sentence."}]}]
pin=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda")
plen=pin["input_ids"].shape[1]
gen=model.generate(**pin,max_new_tokens=24,do_sample=False)
ans=proc.decode(gen[0][plen:],skip_special_tokens=True); print("ANSWER:",repr(ans))
msgs2=msgs+[{"role":"assistant","content":[{"type":"text","text":ans}]}]
fin=proc.apply_chat_template(msgs2,tokenize=True,return_dict=True,return_tensors="pt").to("cuda")
full=fin["input_ids"]; end=full.shape[1]; vmask=(full[0]==IMG)
print("seq",end,"vision",int(vmask.sum().item()),"text",int((~vmask).sum().item()))
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.]); FP8MAX=448.0
def e2(m): lv=E2M1.to(m.device); md=(lv[1:]+lv[:-1])/2; return lv[torch.bucketize(m.clamp(max=6.),md)]
def qdq(x,g):
    D=x.shape[-1]; L=x.shape[:-1]; xb=x.reshape(*L,D//16,16).float()
    ba=xb.abs().amax(-1,keepdim=True); ta=x.float().abs().amax().clamp(min=1e-20)
    gs=((ta/6/FP8MAX)*g).clamp(min=1e-30); sf=((ba/6)/gs).to(torch.float8_e4m3fn).float()
    bs=(sf*gs).clamp(min=1e-30); return (e2(xb.abs()/bs)*xb.sign()*bs).reshape(*L,D).to(x.dtype)
_o=torch.nn.functional.scaled_dot_product_attention; MODE=[None]; S={"vk":[],"vv":[],"tk":[],"tv":[]}
def patched(q,k,v,*a,**kw):
    if MODE[0]=="stats" and k.shape[2]==end:
        if vmask.any(): S["vk"].append(k[:,:,vmask,:].abs().amax().item()); S["vv"].append(v[:,:,vmask,:].abs().amax().item())
        if (~vmask).any(): S["tk"].append(k[:,:,~vmask,:].abs().amax().item()); S["tv"].append(v[:,:,~vmask,:].abs().amax().item())
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
vk=st.mean(S["vk"]); vv=st.mean(S["vv"]); tkk=st.mean(S["tk"]); tvv=st.mean(S["tv"])
print("KVAMAX visK %.3f visV %.3f txtK %.3f txtV %.3f"%(vk,vv,tkk,tvv))
print("RATIO visionOverText K %.2fx V %.2fx"%(vk/tkk,vv/tvv))
b=nll(None); print("ANSWERNLL bf16 %.4f | qdq_g1_calib %+.4f | qdq_g0.5_under %+.4f"%(b,nll(1.0)-b,nll(0.5)-b))
