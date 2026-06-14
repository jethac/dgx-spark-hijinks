# Achievable fp8 D512 dispatcher fix: since the 1-byte D512/VO256 shape exceeds GB10 smem at every
# tiling (proven: cta=64 -> NUM_MMA_KV=1 invalid; cta=16 -> max_mma_kv=0), convert the cryptic
# "please file an issue" crash into a CLEAN, actionable rejection. Inserted before the ragged (3059)
# and paged (3201) DISPATCH_NUM_MMA_KV sites.
p="/work/third_party/flashinfer/include/flashinfer/attention/prefill.cuh"
s=open(p).read()
needle="  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {"
clean='''  {
    const uint32_t _fit = min(max_num_mma_kv_smem, max_num_mma_kv_reg);
    if (_fit < kMinValidMmaKV) {
      std::ostringstream _e;
      _e << "FlashInfer: head_dim_qk=" << HEAD_DIM_QK << " head_dim_vo=" << HEAD_DIM_VO
         << " with " << sizeof(DTypeKV) << "-byte KV exceeds this GPU's shared memory per SM "
         << "(fitted NUM_MMA_KV=" << _fit << " < required " << kMinValidMmaKV
         << "). Use nvfp4 KV (4-bit, ~half the footprint, which fits) or a smaller head_dim.";
      FLASHINFER_ERROR(_e.str());
    }
  }
'''
import re
occs=[m.start() for m in re.finditer(re.escape(needle), s)]
print("dispatch sites:", len(occs))
# insert before the ragged (idx 1) and paged (idx 2) sites, back-to-front to keep offsets
for idx in sorted([1,2], reverse=True):
    if idx < len(occs):
        at=occs[idx]
        s=s[:at]+clean+s[at:]
# ensure <sstream>
if "#include <sstream>" not in s:
    s=s.replace("#include <cstdlib>","#include <cstdlib>\n#include <sstream>",1)
open(p,"w").write(s)
print("PATCHED prefill.cuh (clean reject)")
