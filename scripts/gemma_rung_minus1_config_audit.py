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
        "operator_architecture_hint": "encoder_based_multimodal_text_only_quarantines_vision",
    },
    "gemma4_12b": {
        "model_id": "google/gemma-4-12B-it",
        "rung": "rung_4_encoder_free_multimodal_kv_destination",
        "expected_role": "Gemma 4 12B encoder-free multimodal KV destination",
        "operator_architecture_hint": "encoder_free_multimodal_fused_into_decoder_kv",
    },
    "gemma4_31b": {
        "model_id": "google/gemma-4-31B-it",
        "rung": "rung_2_dense_d512_text_only",
        "expected_role": "dense D=512 isolation rung if config confirms it",
        "operator_architecture_hint": "encoder_based_multimodal_text_only_quarantines_vision",
    },
    "gemma4_26b_a4b": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "rung": "rung_3_moe_text_only",
        "expected_role": "MoE rung after dense D=512 is understood",
        "operator_architecture_hint": "encoder_based_multimodal_text_only_quarantines_vision",
    },
}

VARIANT_MODELS = {
    "gemma4_12b_qat_q4_0_unquantized": {
        "model_id": "google/gemma-4-12B-it-qat-q4_0-unquantized",
        "rung": "rung_4_gemma4_12b_qat_unquantized_probe",
        "expected_role": "Gemma 4 12B QAT-unquantized compatibility check",
        "variant_of": "gemma4_12b",
        "operator_architecture_hint": "encoder_free_multimodal_fused_into_decoder_kv",
    },
    "gemma4_12b_qat_w4a16_ct": {
        "model_id": "google/gemma-4-12B-it-qat-w4a16-ct",
        "rung": "rung_4_gemma4_12b_qat_w4a16_probe",
        "expected_role": "Gemma 4 12B W4A16 checkpoint geometry check",
        "variant_of": "gemma4_12b",
        "operator_architecture_hint": "encoder_free_multimodal_fused_into_decoder_kv",
    },
    "gemma4_31b_qat_w4a16_ct": {
        "model_id": "google/gemma-4-31B-it-qat-w4a16-ct",
        "rung": "rung_2_gemma4_31b_qat_w4a16_probe",
        "expected_role": "Gemma 4 31B W4A16 checkpoint geometry check",
        "variant_of": "gemma4_31b",
        "operator_architecture_hint": "encoder_based_multimodal_text_only_quarantines_vision",
    },
    "gemma4_26b_a4b_qat_q4_0_unquantized": {
        "model_id": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "rung": "rung_3_gemma4_26b_a4b_qat_unquantized_probe",
        "expected_role": "Gemma 4 26B-A4B QAT-unquantized compatibility check",
        "variant_of": "gemma4_26b_a4b",
        "operator_architecture_hint": "encoder_based_multimodal_text_only_quarantines_vision",
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
    bf16_bytes_total = sum(
        row["bf16_raw_kv_bytes_per_token"] or 0 for row in layers
    )
    bf16_bytes_by_layer_type: dict[str, int] = {}
    for row in layers:
        layer_type = row["layer_type"]
        bf16_bytes_by_layer_type[layer_type] = (
            bf16_bytes_by_layer_type.get(layer_type, 0)
            + (row["bf16_raw_kv_bytes_per_token"] or 0)
        )
    ple_like_keys = [
        key
        for key in (
            "hidden_size_per_layer_input",
            "vocab_size_per_layer_input",
            "num_kv_shared_layers",
        )
        if key in norm_text
    ]
    has_text_config = isinstance(raw.get("text_config"), dict)
    has_vision_config = raw.get("vision_config") is not None
    has_audio_config = raw.get("audio_config") is not None
    if has_text_config and (has_vision_config or has_audio_config):
        config_wrapper_status = "wrapper_with_text_config_and_modality_configs"
    elif has_text_config:
        config_wrapper_status = "wrapper_with_text_config_only"
    elif has_vision_config or has_audio_config:
        config_wrapper_status = "root_config_with_modality_configs"
    else:
        config_wrapper_status = "text_only_root_config"
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
        "bf16_raw_kv_bytes_per_token_total": bf16_bytes_total,
        "bf16_raw_kv_bytes_per_token_by_layer_type": bf16_bytes_by_layer_type,
        "has_d512": has_d512,
        "is_moe": is_moe,
        "num_experts": norm_text.get("num_experts"),
        "top_k_experts": norm_text.get("top_k_experts"),
        "moe_intermediate_size": norm_text.get("moe_intermediate_size"),
        "expert_intermediate_size": norm_text.get("expert_intermediate_size"),
        "has_text_config": has_text_config,
        "has_vision_config": has_vision_config,
        "has_audio_config": has_audio_config,
        "config_wrapper_status": config_wrapper_status,
        "is_text_only_root_config": config_wrapper_status == "text_only_root_config",
        "has_ple_like_fields": bool(ple_like_keys),
        "ple_like_keys_present": ple_like_keys,
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
    parser.add_argument(
        "--include-cached-variants",
        action="store_true",
        help="Also audit cached QAT/server checkpoint variants used by the campaign.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {}
    errors: dict[str, str] = {}
    model_specs = dict(DEFAULT_MODELS)
    if args.include_cached_variants:
        model_specs.update(VARIANT_MODELS)

    for key, meta in model_specs.items():
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
                "rung_2_gemma4_31b_dense_d512_text_only",
                "rung_3_gemma4_26b_a4b_moe_text_only",
                "rung_4_gemma4_12b_encoder_free_multimodal_kv",
            ],
            "notes": [
                "Gemma 3 27B config normalizes to uniform D=128, not D=256.",
                "Gemma 4 31B carries full-attention D=512 and is dense, so it can isolate D=512 before MoE.",
                "Gemma 4 26B-A4B carries both D=512 full attention and MoE.",
                "Operator-provided architecture order puts Gemma 4 12B last because its multimodality is fused into the decoder/KV path.",
                "Every serving rung must re-measure this from the running model.",
            ],
        },
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact["decision"], indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
