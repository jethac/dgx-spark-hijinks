import json, math
from types import SimpleNamespace
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolFP4
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

seed=20260611; kv_len=384; qo_len=16; num_qo_heads=32; num_kv_heads=16; head_dim=512; window_left=255; page_size=1; dtype=torch.bfloat16; device='cuda:0'
torch.manual_seed(seed)
fp4_dtype = getattr(torch, 'float4_e2m1fn_x2', torch.uint8)
pool = MHATokenToKVPoolFP4(size=kv_len+8,page_size=page_size,dtype=fp4_dtype,head_num=num_kv_heads,head_dim=head_dim,v_head_dim=head_dim,layer_num=1,device=device,enable_memory_saver=False,start_layer=0,end_layer=1,enable_alt_stream=False)
loc=torch.arange(1,kv_len+1,dtype=torch.int64,device=device)
dense_k=torch.randn(kv_len,num_kv_heads,head_dim,dtype=dtype,device=device)*0.35
dense_v=torch.randn(kv_len,num_kv_heads,head_dim,dtype=dtype,device=device)*0.35
pool.set_kv_buffer(SimpleNamespace(layer_id=0,k_scale=None,v_scale=None,k_scale_float=None,v_scale_float=None), loc, dense_k, dense_v)
q=torch.randn(qo_len,num_qo_heads,head_dim,dtype=dtype,device=device)*0.2
q_indptr=torch.tensor([0,qo_len],dtype=torch.int32,device=device); kv_indptr=torch.tensor([0,kv_len],dtype=torch.int32,device=device); kv_indices=loc.to(torch.int32); kv_last_page_len=torch.tensor([1],dtype=torch.int32,device=device)
k_cache=pool.get_key_buffer(0).unsqueeze(1); v_cache=pool.get_value_buffer(0).unsqueeze(1); k_sf,v_sf=pool.get_kv_scale_buffer(0); k_sf_paged=k_sf.unsqueeze(1); v_sf_paged=v_sf.unsqueeze(1); k_global,v_global=pool.get_kv_global_scale(0)
sm_scale=1/math.sqrt(head_dim)

def run_paged(window_left):
    outs=[]; lses=[]; step=v_cache.shape[-1]//2; sf_step=v_sf_paged.shape[-1]//2
    for i in range(2):
        workspace=torch.empty(256*1024*1024,dtype=torch.uint8,device=device)
        wrapper=BatchPrefillWithPagedKVCacheWrapper(workspace,'NHD',backend='fa2')
        wrapper.plan(q_indptr,kv_indptr,kv_indices,kv_last_page_len,num_qo_heads,num_kv_heads,head_dim,page_size,head_dim_vo=head_dim//2,causal=False,pos_encoding_mode='NONE',sm_scale=sm_scale,window_left=window_left,logits_soft_cap=0.0,q_data_type=dtype,kv_data_type=torch.uint8,k_data_type=torch.uint8,v_data_type=torch.uint8,o_data_type=dtype)
        o,lse=wrapper.run(q,(k_cache,v_cache[...,i*step:(i+1)*step]),k_scale=float(k_global),v_scale=float(v_global),kv_cache_sf=(k_sf_paged,v_sf_paged[...,i*sf_step:(i+1)*sf_step]),return_lse=True)
        outs.append(o); lses.append(lse)
    return torch.cat(outs,dim=-1), lses[0]

def deq(packed, sf, glob):
    return NVFP4KVQuantizeUtil.dequantize(packed.view(torch.uint8), sf.reshape(kv_len,-1), glob, dtype=dtype).reshape(kv_len,num_kv_heads,head_dim).contiguous()
ref_k=deq(k_cache[loc,0], k_sf_paged[loc,0], pool.k_global[0:1]); ref_v=deq(v_cache[loc,0], v_sf_paged[loc,0], pool.v_global[0:1])
out,lse=run_paged(window_left)

def torch_ref(mode):
    group=num_qo_heads//num_kv_heads
    k=ref_k.repeat_interleave(group,dim=1).float(); v=ref_v.repeat_interleave(group,dim=1).float(); qf=q.float()
    scores=torch.einsum('qhd,khd->hqk', qf, k)*sm_scale
    key_pos=torch.arange(kv_len,device=device); query_pos=torch.arange(kv_len-qo_len,kv_len,device=device)
    if mode == 'left_only':
        keep=key_pos[None,:] >= (query_pos[:,None]-window_left)
    elif mode == 'causal_left':
        keep=(key_pos[None,:] >= (query_pos[:,None]-window_left)) & (key_pos[None,:] <= query_pos[:,None])
    elif mode == 'left_plus_one_causal':
        keep=(key_pos[None,:] >= (query_pos[:,None]-(window_left+1))) & (key_pos[None,:] <= query_pos[:,None])
    elif mode == 'same_length_local':
        qpos=torch.arange(qo_len,device=device); keep=(key_pos[None,:] >= (qpos[:,None]-window_left)) & (key_pos[None,:] <= qpos[:,None])
    else:
        raise AssertionError(mode)
    scores=scores.masked_fill(~keep.unsqueeze(0), float('-inf'))
    l=torch.logsumexp(scores,dim=-1).transpose(0,1).contiguous(); p=torch.softmax(scores,dim=-1); r=torch.einsum('hqk,khd->qhd',p,v).to(dtype).contiguous(); return r,l

def cmp(a,b):
    af=a.float().flatten(); bf=b.float().flatten(); return {'cosine':float(torch.nn.functional.cosine_similarity(af,bf,dim=0).item()),'max_abs':float((af-bf).abs().max().item()),'mean_abs':float((af-bf).abs().mean().item())}
res={}
for mode in ['left_only','causal_left','left_plus_one_causal','same_length_local']:
    r,l=torch_ref(mode); res[mode]={'out':cmp(out,r),'lse':cmp(lse,l)}
print(json.dumps(res,indent=2,sort_keys=True))
