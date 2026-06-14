# Verification patch (fp8 path): for asymmetric VO-split (qk>vo) 1-byte KV, prefer cta_tile_q=16
# (NUM_WARPS_Q=2 -> NUM_MMA_KV=1 valid) when the 1-byte 1x4 KV step fits, instead of returning 64
# (NUM_WARPS_Q=4 -> NUM_MMA_KV=1 invalid). Verification-grade (assumes 1-byte for qk>256); the
# production fix threads the real kv_elem_size.
p="/work/third_party/flashinfer/include/flashinfer/utils.cuh"
s=open(p).read()
anchor="const uint32_t qk = head_dim_qk ? head_dim_qk : head_dim;"
inject='''
  { // [FP8 D512 fix verify]
    auto _cc = GetCudaComputeCapability();
    if (_cc.first >= 8 && qk > 256) {
      int _di=0,_ms=0; cudaGetDevice(&_di);
      cudaDeviceGetAttribute(&_ms, cudaDevAttrMaxSharedMemoryPerMultiprocessor, _di);
      if (16u*qk*2u + (qk+head_dim)*16u*4u*1u <= (uint32_t)_ms) return 16;
    }
  }'''
assert anchor in s, "anchor not found"
open(p,"w").write(s.replace(anchor, anchor+inject, 1))
print("PATCHED utils.cuh")
