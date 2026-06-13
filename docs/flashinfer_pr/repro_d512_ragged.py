import torch, flashinfer, sys
dev="cuda"; torch.manual_seed(0)
hq, hv = 512, 256
nq, nk = 8, 1
kv_len = int(sys.argv[1]) if len(sys.argv)>1 else 8192
ws=torch.empty(384*1024*1024,dtype=torch.uint8,device=dev)
def trial(qo_len, kvdt):
    qi=torch.tensor([0,qo_len],dtype=torch.int32,device=dev)
    ki=torch.tensor([0,kv_len],dtype=torch.int32,device=dev)
    q=torch.randn(qo_len,nq,hq,dtype=torch.bfloat16,device=dev)
    k=(torch.randn(kv_len,nk,hq,dtype=torch.bfloat16,device=dev)*0.1).to(kvdt)
    v=(torch.randn(kv_len,nk,hv,dtype=torch.bfloat16,device=dev)*0.1).to(kvdt)
    w=flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws,"NHD")
    w.plan(qi,ki,nq,nk,hq,head_dim_vo=hv,causal=True,
           q_data_type=torch.bfloat16,kv_data_type=kvdt)
    o=w.run(q,k,v); torch.cuda.synchronize(); return tuple(o.shape)
print(f"=== RAGGED head_dim_qk={hq} head_dim_vo={hv} kv_len={kv_len} ===")
for qo in [8,17,32,64,128,256]:
    cta = 64 if qo>16 else 16
    try:
        sh=trial(qo, torch.float8_e4m3fn)
        print(f"  qo={qo:4d} cta_tile_q={cta:3d} NUM_WARPS_Q={4 if cta>16 else 1} fp8: OK {sh}")
    except Exception as e:
        h=[l for l in str(e).splitlines() if "Invalid" in l or "max_mma_kv" in l or "Unsupported" in l]
        print(f"  qo={qo:4d} cta_tile_q={cta:3d} NUM_WARPS_Q={4 if cta>16 else 1} fp8: *** {h[:1] or str(e).splitlines()[-1:]}")
