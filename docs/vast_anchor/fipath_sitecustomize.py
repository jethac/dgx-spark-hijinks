"""sitecustomize: monkeypatch flashinfer prefill wrappers to log which path + module each
attention call uses. Put this dir first on PYTHONPATH so it loads in the EngineCore subprocess.
Reveals whether nvfp4 single (8185) vs chunked (4096) take the same kernel or different paths
(e.g. monolithic paged-nvfp4 vs cascade paged-nvfp4 + ragged-bf16)."""
import sys


def _patch():
    try:
        import flashinfer.prefill as fp
    except Exception as e:
        sys.stderr.write(f"[FIPATH] import fail {e}\n")
        return

    def _modname(self):
        for attr in ("_cached_module", "_jit_module", "_module"):
            m = getattr(self, attr, None)
            if m is not None:
                return getattr(m, "__name__", type(m).__name__)
        return "?"

    def _qolen(self):
        for attr in ("_qo_indptr_buf", "_qo_indptr", "_paged_kv_indptr_buf"):
            t = getattr(self, attr, None)
            try:
                if t is not None:
                    return int(t[-1].item())
            except Exception:
                pass
        return -1

    def wrap(cls, tag):
        if cls is None or not hasattr(cls, "run"):
            return
        orig = cls.run

        def run(self, *a, **k):
            try:
                sys.stderr.write(
                    f"[FIPATH] {tag} qo_tokens={_qolen(self)} module={_modname(self)} "
                    f"return_lse={k.get('return_lse')}\n"
                )
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[FIPATH] {tag} introspect_err={e}\n")
            return orig(self, *a, **k)

        cls.run = run

    wrap(getattr(fp, "BatchPrefillWithPagedKVCacheWrapper", None), "PAGED")
    wrap(getattr(fp, "BatchPrefillWithRaggedKVCacheWrapper", None), "RAGGED")
    sys.stderr.write("[FIPATH] patched\n")


_patch()
