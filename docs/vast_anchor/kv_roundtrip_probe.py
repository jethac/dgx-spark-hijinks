#!/usr/bin/env python3
"""Decisive A-vs-B test for the 26B-A4B nvfp4 break: capture the model's REAL K/V projections and
measure the MINIMUM ACHIEVABLE nvfp4 round-trip error (format loss, kernel-free). If 26B's best-case
format error is comparable to the 12B's (which serves nvfp4 fine), the 4-bit format CAN hold 26B's K/V
-> the serving break is a KERNEL bug (B, fixable). If 26B's is far larger, it's a genuine 4-bit
quantizability limit (A). nvfp4 blocks are 16 contiguous features along head_dim (256/512, both /16),
so reshaping k_proj/v_proj output [seq, feat] and round-tripping with block=16 is the correct format sim.
(k_proj is pre-rope; rope is a per-pair rotation that preserves per-element magnitude, so the per-block
amax structure -- what nvfp4 quantizes against -- is preserved. A small pre-rope error cannot become a
large post-rope one.)
Usage: kv_roundtrip_probe.py <model> <corpus> [nseq=2048]"""
import sys, json, statistics as st, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "/root")
from nvfp4_kv_calibrate import roundtrip_err, best_global_scale

model, corpus = sys.argv[1], sys.argv[2]
nseq = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ids = tok.encode(open(corpus, encoding="utf-8", errors="replace").read(), add_special_tokens=False)[:nseq]
m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda",
                                         trust_remote_code=True, attn_implementation="eager").eval()
caps = {}
def mk(name):
    def h(mod, inp, out): caps[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return h
hooks = [mod.register_forward_hook(mk(n)) for n, mod in m.named_modules()
         if (n.endswith("k_proj") or n.endswith("v_proj")) and "vision" not in n and "audio" not in n]
with torch.no_grad():
    m(input_ids=torch.tensor([ids], device="cuda"))

rows = []
for name, x in caps.items():
    x = x.reshape(-1, x.shape[-1]).float()
    kind = "K" if "k_proj" in name else "V"
    gs, e_best, _ = best_global_scale(x)
    rows.append({"layer": name, "kind": kind, "feat": int(x.shape[-1]),
                 "rt_best": round(e_best, 4), "best_gs": round(gs, 3),
                 "rt_gs0.1": round(roundtrip_err(x, 0.1), 4), "amax": round(float(x.abs().max()), 2)})
tag = model.split("/")[-1]
for kind in ["K", "V"]:
    es = [r["rt_best"] for r in rows if r["kind"] == kind]
    if es:
        print(f"AGG {tag} {kind}: best_rt_rel_l2 median={st.median(es):.4f} mean={sum(es)/len(es):.4f} "
              f"max={max(es):.4f} n={len(es)}")
print(f"WORST {tag}: " + json.dumps(sorted(rows, key=lambda r: -r["rt_best"])[:6]))
print(f"DONE_PROBE {tag}")
