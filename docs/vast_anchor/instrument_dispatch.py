#!/usr/bin/env python3
"""Insert an env-gated kernel-side log of the chosen tiling into the paged-prefill dispatch,
so we can see whether single (8185 qo) vs chunked (4096 qo) pick a different CTA_TILE_Q /
NUM_WARPS_Q / NUM_MMA_KV (the suspected source of the +0.42-vs-+0.19 nvfp4 gap)."""
import re, pathlib
p = pathlib.Path("/root/flashinfer/include/flashinfer/attention/prefill.cuh")
s = p.read_text()
needle = "  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {"
log = (
    '  if (getenv("FI_LOG_DISPATCH")) {\n'
    '    fprintf(stderr, "[FI_DISPATCH] CTA_TILE_Q=%u NUM_WARPS_Q=%u fitted_mma_kv=%u "\n'
    '            "HEAD_DIM_QK=%u HEAD_DIM_VO=%u kvbytes=%u num_qo_heads=%u padded_bs=%u\\n",\n'
    '            (unsigned)CTA_TILE_Q, (unsigned)NUM_WARPS_Q,\n'
    '            (unsigned)min(max_num_mma_kv_smem, max_num_mma_kv_reg),\n'
    '            (unsigned)HEAD_DIM_QK, (unsigned)HEAD_DIM_VO,\n'
    '            (unsigned)sizeof(typename Params::DTypeKV), (unsigned)num_qo_heads,\n'
    '            (unsigned)padded_batch_size);\n'
    '  }\n'
)
# Instrument only the paged dispatch (3rd occurrence); ragged=2nd, single=1st.
occs = [m.start() for m in re.finditer(re.escape(needle), s)]
print("occurrences:", len(occs))
if len(occs) >= 3:
    idx = occs[2]
    s = s[:idx] + log + s[idx:]
    p.write_text(s)
    print("instrumented paged dispatch at offset", idx)
else:
    print("FAILED: expected >=3 occurrences")
# ensure <cstdio>/<cstdlib> available
if "#include <cstdio>" not in s:
    s2 = p.read_text().replace("#include <cstdlib>", "#include <cstdlib>\n#include <cstdio>", 1)
    p.write_text(s2)
