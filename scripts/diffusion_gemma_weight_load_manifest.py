#!/usr/bin/env python3
"""Exercise DiffusionGemma's SGLang weight loader with metadata tensors.

This is the DG-S2 gate between name-remap metadata and a real BF16 serving load.
It instantiates the SGLang model on the meta device, builds fake tensors from
safetensors shard headers, runs the real ``load_weights`` implementation, and
records the loader manifest. Large tensors stay on the meta device; scalar
tensors use tiny CPU zeros because SGLang's default scalar loader calls
``.item()``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - audit should report parse failures.
        return None, repr(exc)
    if not isinstance(data, dict):
        return None, "top-level JSON is not an object"
    return data, None


def resolve_model_dir(
    model_path: str,
    revision: str | None,
    allow_download: bool,
    max_workers: int,
) -> tuple[Path, list[str]]:
    path = Path(model_path)
    warnings: list[str] = []
    if path.exists():
        return path, warnings
    if not allow_download:
        raise RuntimeError(
            f"{model_path!r} is not a local path; pass --allow-download for HF repos"
        )

    from huggingface_hub import snapshot_download  # type: ignore

    warnings.append("model path is a HF repo id; snapshot_download may fetch full shards")
    downloaded = snapshot_download(
        repo_id=model_path,
        revision=revision,
        max_workers=max_workers,
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


def read_safetensors_headers(model_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    from safetensors import safe_open  # type: ignore

    headers: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    entry: dict[str, Any] = {"shard": path.name}
                    try:
                        tensor_slice = handle.get_slice(key)
                        entry["shape"] = list(tensor_slice.get_shape())
                    except Exception as exc:  # noqa: BLE001 - continue auditing.
                        entry["shape_error"] = repr(exc)
                    headers[str(key)] = entry
        except Exception as exc:  # noqa: BLE001 - record corrupt/incompatible shards.
            errors.append(f"{path.name}: {exc!r}")
    return headers, errors


def fake_weight_iter(headers: dict[str, dict[str, Any]]) -> Iterable[tuple[str, Any]]:
    import torch

    for key in sorted(headers):
        shape = headers[key].get("shape")
        if not isinstance(shape, list):
            continue
        shape_tuple = tuple(int(dim) for dim in shape)
        if shape_tuple == () or (len(shape_tuple) == 1 and shape_tuple[0] == 1):
            yield key, torch.zeros(shape_tuple, dtype=torch.bfloat16)
        else:
            yield key, torch.empty(shape_tuple, dtype=torch.bfloat16, device="meta")


def bootstrap_sglang(model_path: str) -> str | None:
    try:
        import torch.distributed as dist
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
    except Exception as exc:  # noqa: BLE001 - caller records this.
        return repr(exc)

    try:
        set_global_server_args_for_scheduler(ServerArgs(model_path=model_path))
        if not dist.is_initialized():
            port = free_port()
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=0,
                backend="gloo",
            )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    except Exception as exc:  # noqa: BLE001 - manifest should preserve failure.
        return repr(exc)
    return None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local HF dir or repo id.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", help="Write manifest JSON to this path.")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    model_dir, warnings = resolve_model_dir(
        args.model_path,
        revision=args.revision,
        allow_download=args.allow_download,
        max_workers=args.max_workers,
    )
    index_map, index_errors = read_index(model_dir)
    headers, header_errors = read_safetensors_headers(model_dir)

    bootstrap_error = bootstrap_sglang(str(model_dir))
    load_error = None
    loaded_params = []
    loader_manifest: dict[str, Any] = {}
    model_class = None
    resolved_arch = None

    try:
        import torch

        from sglang.srt.models.registry import ModelRegistry
        from sglang.srt.utils.hf_transformers_utils import get_config

        config = get_config(str(model_dir), trust_remote_code=True, revision=args.revision)
        architectures = list(getattr(config, "architectures", []) or [])
        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(architectures)
        model_class = model_cls.__name__

        if bootstrap_error is None:
            with torch.device("meta"):
                model = model_cls(config=config, quant_config=None)
            loaded = model.load_weights(fake_weight_iter(headers))
            loaded_params = sorted(str(name) for name in loaded)
            loader_manifest = getattr(model, "dg_weight_load_manifest", {}) or {}
    except Exception as exc:  # noqa: BLE001 - this is the diagnostic payload.
        load_error = repr(exc)

    result: dict[str, Any] = {
        "schema": "diffusion-gemma-weight-load-manifest/v1",
        "model_path": args.model_path,
        "resolved_model_dir": model_dir.as_posix(),
        "revision": args.revision,
        "pid": os.getpid(),
        "warnings": warnings,
        "index_key_count": len(index_map),
        "header_key_count": len(headers),
        "index_errors": index_errors,
        "header_errors": header_errors,
        "bootstrap_error": bootstrap_error,
        "load_error": load_error,
        "resolved_architecture": resolved_arch,
        "model_class": model_class,
        "loaded_param_count": len(loaded_params),
        "loaded_param_sample": loaded_params[:50],
        "loader_manifest": loader_manifest,
    }
    result["ok_for_dg_s2_weight_load_manifest"] = (
        bootstrap_error is None
        and load_error is None
        and bool(headers)
        and bool(loader_manifest)
        and not loader_manifest.get("self_conditioning_missing")
    )

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")

    if args.strict and not result["ok_for_dg_s2_weight_load_manifest"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
