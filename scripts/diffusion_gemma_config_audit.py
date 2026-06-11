#!/usr/bin/env python3
"""Emit the DG-S0 DiffusionGemma config/geometry manifest.

This is intentionally metadata-only: it loads the HF config, resolves SGLang's
model class, instantiates the model on the meta device, and prints the geometry
manifest. It does not download or load model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.models.registry import ModelRegistry
from sglang.srt.utils.hf_transformers_utils import get_config, get_hf_text_config


def geometry_from_config(config) -> dict:
    text_config = get_hf_text_config(config)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    num_layers = getattr(text_config, "num_hidden_layers", len(layer_types))
    if not layer_types:
        layer_types = ["sliding_attention"] * num_layers

    raw_head_dim = getattr(text_config, "head_dim", None)
    local_head_dim = getattr(text_config, "swa_head_dim", None) or raw_head_dim
    global_head_dim = getattr(text_config, "global_head_dim", None) or raw_head_dim
    raw_kv_heads = getattr(text_config, "num_key_value_heads", None)
    local_kv_heads = getattr(text_config, "swa_num_key_value_heads", None) or raw_kv_heads
    global_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or raw_kv_heads
    dtype_bytes = 2

    layers = []
    for layer_id, layer_type in enumerate(layer_types):
        is_global = layer_type == "full_attention"
        head_dim = global_head_dim if is_global else local_head_dim
        kv_heads = global_kv_heads if is_global else local_kv_heads
        layers.append(
            {
                "layer": layer_id,
                "type": layer_type,
                "head_dim": head_dim,
                "num_attention_heads": getattr(text_config, "num_attention_heads", None),
                "num_key_value_heads": kv_heads,
                "sliding_window": None
                if is_global
                else getattr(text_config, "sliding_window", None),
                "bf16_kv_bytes_per_token": (
                    2 * int(kv_heads) * int(head_dim) * dtype_bytes
                    if kv_heads is not None and head_dim is not None
                    else None
                ),
            }
        )

    return {
        "architecture": "DiffusionGemmaForBlockDiffusion",
        "source": "hf_config",
        "canvas_length": getattr(config, "canvas_length", None),
        "max_denoising_steps": getattr(config, "max_denoising_steps", None),
        "confidence_threshold": getattr(config, "confidence_threshold", None),
        "stability_threshold": getattr(config, "stability_threshold", None),
        "sampler_config": getattr(config, "sampler_config", None),
        "num_hidden_layers": num_layers,
        "local_head_dim": local_head_dim,
        "global_head_dim": global_head_dim,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = get_config(
        args.model_path,
        trust_remote_code=True,
        revision=args.revision,
    )
    text_config = get_hf_text_config(config)
    architectures = list(getattr(config, "architectures", []) or [])
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)

    instantiation_error = None
    geometry = None
    try:
        with torch.device("meta"):
            model = model_cls(config=config, quant_config=None)
        geometry = model.get_diffusion_gemma_geometry_manifest()
        geometry["source"] = "sglang_model_meta"
    except Exception as exc:  # noqa: BLE001 - audit should preserve import/init failures.
        instantiation_error = repr(exc)
        geometry = geometry_from_config(config)

    manifest = {
        "model_path": args.model_path,
        "revision": args.revision,
        "architectures": architectures,
        "resolved_architecture": resolved_arch,
        "model_class": model_cls.__name__,
        "model_instantiation_error": instantiation_error,
        "hf_model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "geometry": geometry,
    }

    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
