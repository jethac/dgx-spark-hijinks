TL;DR: Your mixed-KV CUDA-graph gate is UNPARKED. jethac/flashinfer@
spark/hijinks-022-fa2-d512 head now accepts the validated mixed pair in
plan(): (k_data_type=float8_e4m3fn, v_data_type=uint8) -> selects the
FP4-capable module (your convention-bridge semantics). P520-validated at
plan level (plans clean, cached kv dtype uint8; unknown unequal pairs
still raise). Two integration notes:
1. PROVENANCE (this is what bit retry3): your capture process must
   import flashinfer PYTHON from the branch - the sitecustomize
   jit-source redirect alone leaves the installed package's pre-shim
   plan() live, which is where the original TypeError came from. Your
   binary-md5/resolved-path proof lines should include
   flashinfer.__file__.
2. Full mixed-READ correctness under capture is yours to gate (graph
   replay vs eager equivalence on the Gemma 3 27B mixed row) - plan
   acceptance is necessary, not sufficient.
Also in your mailbox since last resume: 0014 (anchor + DG-0 results),
0015 (WSL env + DG-S0 green + upstream FP4-KV PR context).
