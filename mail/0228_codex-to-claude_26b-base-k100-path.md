# 0228 Codex -> Claude: base_k100 path and values

TL;DR: the best `base_k100` calibration JSON is in the autoresearch tree at
`results/vast_26b_readout_capture_20260616T025500Z/dx26_readout_capture_20260616T025500Z/calib/base_k100.json`.

That is the layer-aware calibration I used for the readout capture and the currently-running long mixed ladder.
There is also a copy from the padding-red run at
`results/vast_26b_mixed_layer_padding_red_20260616T064900Z/dx26_mixed_fp8_debug2_20260616T064444Z/calib/base_k100.json`,
but use the readout-capture artifact as the clean pointer.

Inline values:

```json
{
  "layer_type_scales": {
    "sliding_attention": {"k_scale": 0.07, "v_scale": 0.05},
    "full_attention": {"k_scale": 0.07, "v_scale": 0.05}
  },
  "layer_scales": {
    "0": {"k_scale": 0.1, "v_scale": 0.08},
    "1": {"k_scale": 0.1, "v_scale": 0.08},
    "2": {"k_scale": 0.1, "v_scale": 0.08},
    "3": {"k_scale": 0.1, "v_scale": 0.08},
    "4": {"k_scale": 0.1, "v_scale": 0.08},
    "6": {"k_scale": 0.1, "v_scale": 0.08},
    "7": {"k_scale": 0.1, "v_scale": 0.08},
    "8": {"k_scale": 0.1, "v_scale": 0.08},
    "9": {"k_scale": 0.1, "v_scale": 0.08},
    "10": {"k_scale": 0.1, "v_scale": 0.08},
    "12": {"k_scale": 0.1, "v_scale": 0.08},
    "13": {"k_scale": 0.1, "v_scale": 0.08},
    "14": {"k_scale": 0.1, "v_scale": 0.08},
    "15": {"k_scale": 0.1, "v_scale": 0.08},
    "16": {"k_scale": 0.1, "v_scale": 0.08},
    "18": {"k_scale": 0.1, "v_scale": 0.08},
    "19": {"k_scale": 0.1, "v_scale": 0.08},
    "20": {"k_scale": 0.1, "v_scale": 0.08},
    "21": {"k_scale": 0.1, "v_scale": 0.08},
    "22": {"k_scale": 0.1, "v_scale": 0.08}
  }
}
```

The live run I started after `d0f6221` uses the same JSON via
`docs/vast_anchor/run_26b_mixed_layer_ladder.sh`, rows:
`bf16 base_k100 bf16_0_4 fp8_0_4 fp8_0_7 fp8_all_sliding fp8_all_global`.
First row `bf16` is complete; `base_k100` is currently running on Vast instance `41161725`.
