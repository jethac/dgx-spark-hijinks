# External references

## Saiyam Pathak (Kubesimplify) - DGX Spark series, day 3 (2026-06-05)
https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists

Independent validation of the campaign's framing, at the layer above ours
(Ollama/NIM/TRT-LLM; explicitly does NOT cover vLLM/SGLang/FlashInfer):
- sm_121-vs-sm_100 late-support framing + source-rebuild workaround = our
  blog's BEFORE narrative, third-party. CITE in the blog opening.
- Bandwidth-bound decode math (273 GB/s ceiling) = the capacity-not-speed
  thesis, independently arrived at.
- DATAPOINT for the future 26B rung row: Gemma 4 26B MoE ~65.6 tok/s on his
  stack (compare when our row lands).
- Practical corroborations: nvidia-smi reports [N/A] under ATS unified
  memory (use free -h); 7-part series in progress June 2026 - author is a
  DevRel-to-DevRel amplification contact when the blog ships (Jetha's call
  on timing).
