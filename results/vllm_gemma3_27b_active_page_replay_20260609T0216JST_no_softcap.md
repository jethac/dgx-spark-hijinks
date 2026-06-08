# vLLM Active-Page Replay

- logits_soft_cap: `0.0`
- sm_scale input: `None`
- deswizzle_v_scale: `False`

## Dumps

### spark_active_page_prefill_language_model_model_layers_5_self_attn_attn_0001.pt

- prefill/decode tokens: `8` / `8`
- active pages: `[0]`
- paged indices: `[0, 0, 0, 0, 0, 0, 0, 0, 0]`
- reference stats: min `0.0`, max `0.0`, mean `0.0`, rms `0.0`
- out_after stats: min `0.0`, max `0.0`, mean `0.0`, rms `0.0`, byte_like `False`
- out_after vs reference: cosine `0.0`, max_abs `0.0`, mean_abs `0.0`
- out_after vs repeated active V bytes: cosine `None`, max_abs `None`, mean_abs `None`
- reference head: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- out_after head: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

### spark_active_page_prefill_language_model_model_layers_5_self_attn_attn_0002.pt

- prefill/decode tokens: `18` / `0`
- active pages: `[13, 14]`
- paged indices: `[13, 14]`
- reference stats: min `-14.749245643615723`, max `15.0`, mean `-0.008165801875293255`, rms `1.9468220472335815`
- out_after stats: min `0.0`, max `255.0`, mean `128.52850341796875`, rms `147.44287109375`, byte_like `True`
- out_after vs reference: cosine `-0.0042778667993843555`, max_abs `259.5`, mean_abs `128.55294799804688`
- out_after vs repeated active V bytes: cosine `0.9488214254379272`, max_abs `242.0`, mean_abs `25.90928840637207`
- reference head: `[0.0, -4.875, 0.40625, 0.0, 0.8125, -3.25, -0.40625, 0.0, 0.40625, -0.40625, 1.625, 0.40625, -0.8125, -1.21875, -0.40625, -1.21875]`
- out_after head: `[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, 33.0, 65.0, 47.0, 233.0, 91.0, 34.0, 145.0, 25.0]`

### spark_active_page_prefill_language_model_model_layers_5_self_attn_attn_0003.pt

- prefill/decode tokens: `23` / `0`
- active pages: `[21, 22]`
- paged indices: `[21, 22]`
- reference stats: min `-13.5`, max `15.0`, mean `-0.019330691546201706`, rms `1.919629454612732`
- out_after stats: min `0.0`, max `255.0`, mean `129.38995361328125`, rms `148.07107543945312`, byte_like `True`
- out_after vs reference: cosine `-0.002427698578685522`, max_abs `259.5`, mean_abs `129.41940307617188`
- out_after vs repeated active V bytes: cosine `0.9488015174865723`, max_abs `243.0`, mean_abs `26.28031349182129`
- reference head: `[0.0, -4.875, 0.40625, 0.0, 0.8125, -3.25, -0.40625, 0.0, 0.40625, -0.40625, 1.625, 0.40625, -0.8125, -1.21875, -0.40625, -1.21875]`
- out_after head: `[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, 33.0, 65.0, 47.0, 233.0, 91.0, 34.0, 145.0, 25.0]`
