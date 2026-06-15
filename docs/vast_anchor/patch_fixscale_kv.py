# Separate-K/V fixed-scale shim (the per-K/V calibration sweep). Sets _k_scale and _v_scale
# INDEPENDENTLY from VLLM_FIX_K_SCALE / VLLM_FIX_V_SCALE (falling back to VLLM_FIX_KV_SCALE for
# either when its specific var is unset). Write- and read-side are kept consistent (both endpoints
# set to the same value) per the decouple matrix. Idempotent, revertible (FIXSCALEKV marker).
import importlib.util, os, py_compile
vs = importlib.util.find_spec("vllm"); VROOT = os.path.dirname(vs.origin)
p = VROOT + "/model_executor/layers/attention/attention.py"
lines = [l for l in open(p).read().split("\n") if "FIXSCALEKV" not in l]
out, done = [], False
for l in lines:
    out.append(l)
    if (not done) and l.strip() == "if self.calculate_kv_scales:":
        ind = l[:len(l) - len(l.lstrip())]
        # inject BEFORE the matched 'if' line (so it runs regardless of calculate_kv_scales)
        for code in [
            "import os as _fk  # FIXSCALEKV",
            "if _fk.environ.get('VLLM_FIX_K_SCALE') or _fk.environ.get('VLLM_FIX_V_SCALE') or _fk.environ.get('VLLM_FIX_KV_SCALE'):  # FIXSCALEKV",
            "    _kv=_fk.environ.get('VLLM_FIX_KV_SCALE'); _k=float(_fk.environ.get('VLLM_FIX_K_SCALE', _kv)); _v=float(_fk.environ.get('VLLM_FIX_V_SCALE', _kv))  # FIXSCALEKV",
            "    self._k_scale.fill_(_k); self._v_scale.fill_(_v); self._k_scale_float=_k; self._v_scale_float=_v  # FIXSCALEKV",
        ]:
            out.insert(len(out) - 1, ind + code)
        done = True
open(p, "w").write("\n".join(out))
py_compile.compile(p, doraise=True)
print("patch_fixscale_kv OK" if done else "MARKER NOT FOUND")
