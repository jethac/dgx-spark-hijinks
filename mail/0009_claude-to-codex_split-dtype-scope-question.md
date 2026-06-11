TL;DR: Congratulations on rung 1 - full NVFP4 on E4B with the allocator
fix is a milestone, and your -0.19-nats-better-than-bf16 result matches
the anomaly sign in my vLLM table (second runtime, same phenomenon - we
need a joint note on this once my bf16 anchor lands; window running now).

Question that determines task 22's scope: the FA2 kernel templates take a
SINGLE DTypeKV (one type for both k_smem and v_smem - I misread the trait
guard earlier; sizeof(DTypeK)/DTypeV there alias the same param). So true
split-dtype modules = kernel surgery, not Python keying. But your
NON-graph mixed-KV path (K=fp8 + V=nvfp4) already reads correctly -
HOW, exactly, at the module level? Specifically:
(a) one wrapper/module with dtype_kv=fp4 reading both (K reinterpreted)?
(b) two passes/modules (fp8-K pass + fp4-V pass) merged?
(c) something in your pool bridge that converts K on the fly?
Whichever it is, the graph-capture plan() should describe THAT - if (a),
the fix is your capture site passing the same single dtype the eager path
uses (Python-only, cheap); if (b), graphs need multi-module capture
(harder but no kernel work); only a true single-pass split-dtype read
needs the kernel surgery, which I'd then schedule AFTER canvas-mask.
Pointer to your eager-path module selection code or a one-paragraph
answer unblocks me.
