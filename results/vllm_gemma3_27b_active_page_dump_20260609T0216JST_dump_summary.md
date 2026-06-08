# vLLM Active-Page Dump Summary

- dump_dir: `results\vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_active_page_dump`
- files: `3`
- byte-like `out_after` tensors: `2`

## Dumps

### spark_active_page_prefill_language_model_model_layers_5_self_attn_attn_0001.pt

- layer: `language_model.model.layers.5.self_attn.attn`
- window_left: `-1`
- prefill/decode tokens: `8` / `8`
- active_pages: `[0]`
- paged_kv_indices_head: `[0, 0, 0, 0, 0, 0, 0, 0, 0]`
- last_page_len: `[9]`
- out_after byte-like: `False`
- out_after stats: min `0.0`, max `0.0`, mean `0.0`, rms `0.0`
- out_after head: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- active V data page head: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- active V scale page head: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

### spark_active_page_prefill_language_model_model_layers_5_self_attn_attn_0002.pt

- layer: `language_model.model.layers.5.self_attn.attn`
- window_left: `-1`
- prefill/decode tokens: `18` / `0`
- active_pages: `[13, 14]`
- paged_kv_indices_head: `[13, 14]`
- last_page_len: `[2]`
- out_after byte-like: `True`
- out_after stats: min `0.0`, max `255.0`, mean `128.52850341796875`, rms `147.44287109375`
- out_after head: `[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, 33.0, 65.0, 47.0, 233.0, 91.0, 34.0, 145.0, 25.0]`
- active V data page head: `[240, 1, 226, 137, 145, 20, 186, 185, 33, 65, 47, 233, 91, 34, 145, 25]`
- active V scale page head: `[0.8125, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.25, 0.34375, 0.28125, 0.28125, 0.75, 0.15625, 0.4375, 0.4375, 1.0]`

### spark_active_page_prefill_language_model_model_layers_5_self_attn_attn_0003.pt

- layer: `language_model.model.layers.5.self_attn.attn`
- window_left: `-1`
- prefill/decode tokens: `23` / `0`
- active_pages: `[21, 22]`
- paged_kv_indices_head: `[21, 22]`
- last_page_len: `[7]`
- out_after byte-like: `True`
- out_after stats: min `0.0`, max `255.0`, mean `129.38995361328125`, rms `148.07107543945312`
- out_after head: `[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, 33.0, 65.0, 47.0, 233.0, 91.0, 34.0, 145.0, 25.0]`
- active V data page head: `[240, 1, 226, 137, 145, 20, 186, 185, 33, 65, 47, 233, 91, 34, 145, 25]`
- active V scale page head: `[0.8125, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.375, 0.34375, 0.28125, 0.28125, 0.75, 0.15625, 0.4375, 0.4375, 1.0]`
