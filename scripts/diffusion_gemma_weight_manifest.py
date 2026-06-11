#!/usr/bin/env python3
"""Audit DiffusionGemma checkpoint names for SGLang's DG-S2 weight remap.

The script is metadata-only. It reads safetensors indexes and, when available,
shard headers for tensor shapes. It does not materialize tensor payloads.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ENCODER_PREFIX = "model.encoder.language_model."
DECODER_PREFIX = "model.decoder."
VISION_PREFIXES = (
    "model.encoder.vision_tower.",
    "model.encoder.embed_vision.",
    "vision_tower.",
    "embed_vision.",
)


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - audit should report parse failures.
        return None, repr(exc)
    if not isinstance(data, dict):
        return None, "top-level JSON is not an object"
    return data, None


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_model_dir(model_path: str, revision: str | None) -> tuple[Path, list[str]]:
    path = Path(model_path)
    warnings: list[str] = []
    if path.exists():
        return path, warnings

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise RuntimeError(
            f"{model_path!r} is not a local path and huggingface_hub is unavailable: {exc!r}"
        ) from exc

    warnings.append(
        "model path is a HF repo id; snapshot_download uses checkpoint metadata patterns"
    )
    downloaded = snapshot_download(
        repo_id=model_path,
        revision=revision,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "*.safetensors.index.json",
            "*.safetensors",
        ],
    )
    return Path(downloaded), warnings


def read_index(model_dir: Path) -> tuple[dict[str, str], list[str]]:
    weight_map: dict[str, str] = {}
    errors: list[str] = []
    for path in sorted(model_dir.glob("*.safetensors.index.json")):
        data, error = load_json(path)
        if error:
            errors.append(f"{path.name}: {error}")
            continue
        raw_map = data.get("weight_map")
        if not isinstance(raw_map, dict):
            errors.append(f"{path.name}: missing object weight_map")
            continue
        for key, shard in raw_map.items():
            weight_map[str(key)] = str(shard)
    return weight_map, errors


def read_safetensors_metadata(model_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        return {}, []

    try:
        from safetensors import safe_open  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        return {}, [f"safetensors unavailable: {exc!r}"]

    tensors: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    info: dict[str, Any] = {"shard": path.name}
                    try:
                        tensor_slice = handle.get_slice(key)
                        info["shape"] = list(tensor_slice.get_shape())
                    except Exception as exc:  # noqa: BLE001 - keep auditing other keys.
                        info["shape_error"] = repr(exc)
                    tensors[str(key)] = info
        except Exception as exc:  # noqa: BLE001 - record corrupt/incompatible shards.
            errors.append(f"{path.name}: {exc!r}")
    return tensors, errors


def remap_backbone_name(name: str) -> str | None:
    if name.startswith(VISION_PREFIXES):
        return None

    if name.startswith(ENCODER_PREFIX):
        suffix = name[len(ENCODER_PREFIX) :]
        return suffix if suffix.startswith(("model.", "lm_head.")) else "model." + suffix

    if name.startswith(DECODER_PREFIX):
        suffix = name[len(DECODER_PREFIX) :]
        if suffix.startswith("self_conditioning."):
            return None
        return suffix if suffix.startswith(("model.", "lm_head.")) else "model." + suffix

    return name


def classify_key(name: str) -> str:
    if name.startswith(VISION_PREFIXES):
        return "vision_quarantined"
    if name.startswith(ENCODER_PREFIX):
        return "encoder_backbone"
    if name.startswith(DECODER_PREFIX + "self_conditioning."):
        return "self_conditioning"
    if name.startswith(DECODER_PREFIX):
        return "decoder_backbone_candidate"
    return "other"


def build_manifest(model_dir: Path, revision: str | None, warnings: list[str]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config, config_error = (
        load_json(model_dir / "config.json") if (model_dir / "config.json").exists() else (None, None)
    )
    generation_config, generation_error = (
        load_json(model_dir / "generation_config.json")
        if (model_dir / "generation_config.json").exists()
        else (None, None)
    )
    index_map, index_errors = read_index(model_dir)
    tensor_meta, safetensors_errors = read_safetensors_metadata(model_dir)

    keys = sorted(set(index_map) | set(tensor_meta))
    categories = Counter(classify_key(key) for key in keys)
    mapped_sources: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        mapped = remap_backbone_name(key)
        if mapped is not None:
            mapped_sources[mapped].append(key)

    duplicates = {
        mapped: sources for mapped, sources in mapped_sources.items() if len(sources) > 1
    }
    duplicate_decoder_sources = {
        mapped: sources
        for mapped, sources in duplicates.items()
        if any(source.startswith(ENCODER_PREFIX) for source in sources)
        and any(source.startswith(DECODER_PREFIX) for source in sources)
    }
    self_conditioning = [key for key in keys if classify_key(key) == "self_conditioning"]
    vision = [key for key in keys if classify_key(key) == "vision_quarantined"]

    shape_samples = {}
    for key in self_conditioning[:20] + vision[:5]:
        if key in tensor_meta:
            shape_samples[key] = tensor_meta[key]

    text_config = (config or {}).get("text_config") if isinstance(config, dict) else None
    if not isinstance(text_config, dict):
        text_config = {}

    result: dict[str, Any] = {
        "schema": "diffusion-gemma-weight-manifest/v1",
        "model_dir": rel(model_dir, repo_root),
        "revision": revision,
        "config": {
            "exists": (model_dir / "config.json").exists(),
            "error": config_error,
            "model_type": (config or {}).get("model_type") if isinstance(config, dict) else None,
            "architectures": (config or {}).get("architectures")
            if isinstance(config, dict)
            else None,
            "text_model_type": text_config.get("model_type"),
            "canvas_length": (config or {}).get("canvas_length") if isinstance(config, dict) else None,
            "max_denoising_steps": (config or {}).get("max_denoising_steps")
            if isinstance(config, dict)
            else None,
            "head_dim": text_config.get("head_dim"),
            "global_head_dim": text_config.get("global_head_dim"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "num_global_key_value_heads": text_config.get("num_global_key_value_heads"),
        },
        "generation_config": {
            "exists": (model_dir / "generation_config.json").exists(),
            "error": generation_error,
            "max_denoising_steps": (generation_config or {}).get("max_denoising_steps")
            if isinstance(generation_config, dict)
            else None,
            "sampler_config": (generation_config or {}).get("sampler_config")
            if isinstance(generation_config, dict)
            else None,
        },
        "key_source": "safetensors" if tensor_meta else "index" if index_map else "none",
        "key_count": len(keys),
        "index_key_count": len(index_map),
        "safetensors_key_count": len(tensor_meta),
        "index_errors": index_errors,
        "safetensors_errors": safetensors_errors,
        "category_counts": dict(sorted(categories.items())),
        "mapped_backbone_key_count": len(mapped_sources),
        "duplicate_mapped_key_count": len(duplicates),
        "decoder_backbone_candidate_key_count": categories["decoder_backbone_candidate"],
        "decoder_duplicate_backbone_key_count": len(duplicate_decoder_sources),
        "self_conditioning_key_count": len(self_conditioning),
        "vision_quarantined_key_count": len(vision),
        "sample_self_conditioning_keys": self_conditioning[:30],
        "sample_vision_keys": vision[:20],
        "sample_duplicate_mapped_keys": {
            key: value for key, value in list(sorted(duplicate_decoder_sources.items()))[:20]
        },
        "sample_shape_metadata": shape_samples,
        "warnings": list(warnings),
    }

    ok = bool(keys) and categories["encoder_backbone"] > 0 and categories["self_conditioning"] > 0
    if categories["vision_quarantined"] == 0:
        result["warnings"].append("no vision keys found to quarantine")
    if categories["decoder_backbone_candidate"] == 0:
        result["warnings"].append("no decoder backbone candidate keys found")
    if not duplicate_decoder_sources:
        result["warnings"].append("no true encoder/decoder duplicate backbone keys found")
    if categories["self_conditioning"] == 0:
        result["warnings"].append("no decoder self-conditioning keys found")
    if not keys:
        result["warnings"].append("no checkpoint keys found")

    result["ok_for_dg_s2_weight_remap_probe"] = ok
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local HF directory or repo id.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", help="Write manifest JSON to this path.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    model_dir, warnings = resolve_model_dir(args.model_path, args.revision)
    result = build_manifest(model_dir, args.revision, warnings)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and not result["ok_for_dg_s2_weight_remap_probe"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
