import torch, flashinfer, sys, traceback
dev="cuda"; torch.manual_seed(0)
head_dim = int(sys.argv[1])
num_qo_heads, num_kv_heads, page_size = 8, 1, 16
kv_len = int(sys.argv[2]) if len(sys.argv)>2 else 8192
num_pages=(kv_len+page_size-1)//page_size
ws=torch.empty(256*1024*1024,dtype=torch.uint8,device=dev)
def trial(qo_len):
    qo_indptr=torch.tensor([0,qo_len],dtype=torch.int32,device=dev)
    kv_indptr=torch.tensor([0,num_pages],dtype=torch.int32,device=dev)
    kv_indices=torch.arange(num_pages,dtype=torch.int32,device=dev)
    last=kv_len-(num_pages-1)*page_size
    klp=torch.tensor([last if last>0 else page_size],dtype=torch.int32,device=dev)
    q=torch.randn(qo_len,num_qo_heads,head_dim,dtype=torch.bfloat16,device=dev)
    kv=(torch.randn(num_pages,2,page_size,num_kv_heads,head_dim,dtype=torch.bfloat16,device=dev)*0.1).to(torch.float8_e4m3fn)
    w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,"NHD")
    w.plan(qo_indptr,kv_indptr,kv_indices,klp,num_qo_heads,num_kv_heads,head_dim,page_size,
           causal=True,q_data_type=torch.bfloat16,kv_data_type=torch.float8_e4m3fn)
    o=w.run(q,kv); torch.cuda.synchronize(); return tuple(o.shape)
print(f"=== head_dim={head_dim} kv_len={kv_len} fp8 KV (sweep qo_len -> cta_tile_q) ===")
for qo in [17,24,32,48,64,96,128,256,512]:
    cta = 128 if (qo>64 and head_dim<256) else (64 if qo>16 else 16)
    try:
        sh=trial(qo); print(f"  qo_len={qo:4d} cta_tile_q={cta:3d} NUM_WARPS_Q={4 if cta>16 else 1}: OK {sh}")
    except Exception as e:
        msg=str(e).splitlines()
        hit=[l for l in msg if "Invalid" in l or "max_mma_kv" in l or "NUM_MMA" in l]
        print(f"  qo_len={qo:4d} cta_tile_q={cta:3d} NUM_WARPS_Q={4 if cta>16 else 1}: *** ERROR: {hit[:1] or msg[-1:]}")
