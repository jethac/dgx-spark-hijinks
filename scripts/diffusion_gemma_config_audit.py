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

    with torch.device("meta"):
        model = model_cls(config=config, quant_config=None)

    manifest = {
        "model_path": args.model_path,
        "revision": args.revision,
        "architectures": architectures,
        "resolved_architecture": resolved_arch,
        "model_class": model_cls.__name__,
        "hf_model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "geometry": model.get_diffusion_gemma_geometry_manifest(),
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

