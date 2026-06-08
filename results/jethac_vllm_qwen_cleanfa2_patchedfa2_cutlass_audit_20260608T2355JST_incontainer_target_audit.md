# In-Container Target Audit: jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST

Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`

Artifacts:

- runtime versions: `/workspace/dgx-spark-hijinks/results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_versions.json`
- CUDA object audit: `/workspace/dgx-spark-hijinks/results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_cuda_so_audit.json`
- CUDA artifact/JIT-cache audit: `/workspace/dgx-spark-hijinks/results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_cuda_artifact_arch_audit.json`

Findings:

- Device: `NVIDIA GB10`, capability `[12, 1]`, SMs `48`.
- Torch: `2.12.0.dev20260408+cu130`, CUDA `13.0`.
- Torch arch list: `['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120', 'compute_120']`.
- Package roots: `{'flashinfer': {'file': '/usr/local/lib/python3.12/dist-packages/flashinfer/__init__.py', 'version': '0.6.9rc1'}, 'vllm': {'file': '/opt/jethac-vllm/vllm/__init__.py', 'version': '0.1.dev1+ga919d635d'}}`.
- Inspected CUDA objects: `13`.
- Architecture counts: sm_100=5, sm_110=3, sm_120=3, sm_121a=1, sm_75=1, sm_80=3, sm_87=3, sm_89=3, sm_90=3, sm_90a=5.
- Objects with `sm_120`: `3`.
- Objects with `sm_121`: `0`.
- Artifact/JIT architecture counts: compute_100f=2, compute_110f=2, compute_120=3, compute_120f=2, compute_80=2, compute_89=2, compute_90=2, sm_100=13, sm_100f=2, sm_103=8, sm_110=11, sm_110f=1, sm_120=11, sm_120f=4, sm_121=8, sm_121a=3, sm_35=8, sm_37=8, sm_50=8, sm_52=8, sm_53=8, sm_60=8, sm_61=8, sm_62=8, sm_70=8, sm_72=8, sm_75=12, sm_80=11, sm_86=8, sm_87=11, sm_88=8, sm_89=11, sm_90=11, sm_90a=5.
- Artifacts with `sm_121`: `8`.
- Artifacts with `sm_121a`: `3`.
- Artifacts with `compute_121`: `0`.
- Artifacts with `compute_121a`: `0`.

Conclusion: inspected CUDA objects include explicit `sm_121`/`sm_121a` target evidence.
