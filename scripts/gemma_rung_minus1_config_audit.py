#!/usr/bin/env python3
"""Audit Gemma config geometry for the compatibility ladder.

This is the cheap Rung -1 pass: it reads model config metadata only. Running
servers still have to re-measure attention geometry before any rung is green.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

try:
    import transformers
    from transformers import AutoConfig
except Exception:  # pragma: no cover - the script records the missing optional dep.
    AutoConfig = None
    transformers = None


DEFAULT_MODELS = {
    "gemma3_27b": {
        "model_id": "google/gemma-3-27b-it",
        "rung": "rung_1_swa_isolation",
        "expected_role": "dense Gemma 3 SWA rung",
    },
    "gemma4_12b": {
        "model_id": "google/gemma-4-12B-it",
        "rung": "rung_2_gemma4_arch",
        "expected_role": "Gemma 4 12B unified architecture rung",
    },
    "gemma4_31b": {
        "model_id": "google/gemma-4-31B-it",
        "rung": "rung_3_dense_d512",
        "expected_role": "dense D=512 isolation rung if config confirms it",
    },
    "gemma4_26b_a4b": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "rung": "rung_4_moe",
        "expected_role": "MoE rung after dense D=512 is understood",
    },
}


def snapshot_from_path(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" not in parts:
        return None
    idx = parts.index("snapshots")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def as_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    return dict(config.__dict__)


def text_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if isinstance(text, dict):
        return text
    return config


def compact_indices(indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {"count": 0, "indices": []}
    return {
        "count": len(indices),
        "indices": indices,
        "first": indices[0],
        "last": indices[-1],
    }


def layer_geometry(text: dict[str, Any]) -> list[dict[str, Any]]:
    layer_types = text.get("layer_types")
    num_layers = text.get("num_hidden_layers")
    if not isinstance(num_layers, int):
        return []
    if not isinstance(layer_types, list):
        layer_types = [None] * num_layers

    head_dim = text.get("head_dim")
    global_head_dim = text.get("global_head_dim")
    kv_heads = text.get("num_key_value_heads")
    global_kv_heads = text.get("num_global_key_value_heads")
    attention_k_eq_v = bool(text.get("attention_k_eq_v"))
    q_heads = text.get("num_attention_heads")
    sliding_window = text.get("sliding_window")

    rows: list[dict[str, Any]] = []
    for idx in range(num_layers):
        layer_type = layer_types[idx] if idx < len(layer_types) else None
        is_full = layer_type == "full_attention"
        resolved_head_dim = global_head_dim if is_full and global_head_dim else head_dim
        resolved_kv_heads = (
            global_kv_heads
            if is_full and attention_k_eq_v and global_kv_heads is not None
            else kv_heads
        )
        bf16_kv_bytes_per_token = None
        if isinstance(resolved_kv_heads, int) and isinstance(resolved_head_dim, int):
            bf16_kv_bytes_per_token = 2 * resolved_kv_heads * resolved_head_dim * 2
        rows.append(
            {
                "layer": idx,
                "layer_type": layer_type or "unknown",
                "head_dim": resolved_head_dim,
                "num_attention_heads": q_heads,
                "num_kv_heads": resolved_kv_heads,
                "sliding_window": sliding_window if layer_type == "sliding_attention" else None,
                "bf16_raw_kv_bytes_per_token": bf16_kv_bytes_per_token,
            }
        )
    return rows


def summarize(model_key: str, model_id: str, raw: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    raw_text = text_config_dict(raw)
    norm_text = text_config_dict(normalized) if normalized else raw_text
    layers = layer_geometry(norm_text)
    full = [row["layer"] for row in layers if row["layer_type"] == "full_attention"]
    sliding = [row["layer"] for row in layers if row["layer_type"] == "sliding_attention"]
    head_dims = sorted({row["head_dim"] for row in layers if row["head_dim"] is not None})
    full_head_dims = sorted(
        {row["head_dim"] for row in layers if row["layer_type"] == "full_attention"}
    )
    sliding_head_dims = sorted(
        {row["head_dim"] for row in layers if row["layer_type"] == "sliding_attention"}
    )
    has_d512 = 512 in head_dims
    is_moe = bool(norm_text.get("enable_moe_block") or norm_text.get("num_experts"))
    return {
        "model_key": model_key,
        "model_id": model_id,
        "root_model_type": normalized.get("model_type", raw.get("model_type")),
        "text_model_type": norm_text.get("model_type"),
        "architectures": raw.get("architectures"),
        "num_hidden_layers": norm_text.get("num_hidden_layers"),
        "num_attention_heads": norm_text.get("num_attention_heads"),
        "num_key_value_heads": norm_text.get("num_key_value_heads"),
        "num_global_key_value_heads": norm_text.get("num_global_key_value_heads"),
        "head_dim": norm_text.get("head_dim"),
        "global_head_dim": norm_text.get("global_head_dim"),
        "sliding_window": norm_text.get("sliding_window"),
        "attention_k_eq_v": norm_text.get("attention_k_eq_v"),
        "layer_type_counts": {
            "sliding_attention": len(sliding),
            "full_attention": len(full),
            "unknown": len([row for row in layers if row["layer_type"] == "unknown"]),
        },
        "full_attention_indices": compact_indices(full),
        "sliding_attention_indices": compact_indices(sliding),
        "head_dims_observed_from_normalized_config": head_dims,
        "sliding_head_dims": sliding_head_dims,
        "full_head_dims": full_head_dims,
        "has_d512": has_d512,
        "is_moe": is_moe,
        "num_experts": norm_text.get("num_experts"),
        "top_k_experts": norm_text.get("top_k_experts"),
        "moe_intermediate_size": norm_text.get("moe_intermediate_size"),
        "expert_intermediate_size": norm_text.get("expert_intermediate_size"),
        "has_vision_config": raw.get("vision_config") is not None,
        "has_audio_config": raw.get("audio_config") is not None,
        "has_ple_like_fields": any(
            key in norm_text
            for key in (
                "hidden_size_per_layer_input",
                "vocab_size_per_layer_input",
                "num_kv_shared_layers",
            )
        ),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="JSON artifact path")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact Hugging Face; use cached config files only.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, meta in DEFAULT_MODELS.items():
        model_id = meta["model_id"]
        try:
            config_path = Path(
                hf_hub_download(
                    model_id,
                    "config.json",
                    local_files_only=args.local_files_only,
                )
            )
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            normalized = raw
            normalized_error = None
            if AutoConfig is not None:
                try:
                    cfg = AutoConfig.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        local_files_only=args.local_files_only,
                    )
                    normalized = as_dict(cfg)
                except Exception as exc:  # pragma: no cover - recorded in output.
                    normalized_error = f"{type(exc).__name__}: {exc}"
            else:
                normalized_error = "transformers is not importable"
            summary = summarize(key, model_id, raw, normalized)
            summary.update(meta)
            summary["config_path"] = str(config_path)
            summary["snapshot"] = snapshot_from_path(config_path)
            summary["normalized_config_error"] = normalized_error
            models[key] = summary
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"

    d512_models = [
        model["model_key"]
        for model in models.values()
        if model.get("has_d512")
    ]
    artifact = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "claim_class": "config_audit_only_running_geometry_still_required",
        "tool_versions": {
            "transformers": getattr(transformers, "__version__", None),
        },
        "models": models,
        "errors": errors,
        "decision": {
            "d512_models": d512_models,
            "does_31b_isolate_d512_dense": "gemma4_31b" in d512_models
            and not models.get("gemma4_31b", {}).get("is_moe", True),
            "does_26b_stack_d512_and_moe": "gemma4_26b_a4b" in d512_models
            and bool(models.get("gemma4_26b_a4b", {}).get("is_moe")),
            "recommended_main_ladder_after_audit": [
                "rung_0_qwen_standard_attention_done",
                "rung_1_gemma3_27b_swa_uniform_d128_no_d512",
                "rung_2_gemma4_12b_unified_arch_plus_d512_audio",
                "rung_3_gemma4_31b_dense_d512",
                "rung_4_gemma4_26b_a4b_moe_on_d512_base",
            ],
            "notes": [
                "Gemma 3 27B config normalizes to uniform D=128, not D=256.",
                "Gemma 4 31B carries full-attention D=512 and is dense, so it can isolate D=512 before MoE.",
                "Gemma 4 26B-A4B carries both D=512 full attention and MoE.",
                "Every serving rung must re-measure this from the running model.",
            ],
        },
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact["decision"], indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
