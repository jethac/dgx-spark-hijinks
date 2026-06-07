#!/usr/bin/env python3
"""Patch an installed SGLang site-package for the FP4 KV overlay probe.

This is an experiment helper, not a production install path. It applies the
minimal stock-image edits needed to test whether the jethac SGLang FP4 KV fixes
move NVIDIA's 26.05 container past the observed Qwen fp4_e2m1 startup blockers.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"could not find patch anchor for {label}")
    return text.replace(old, new, 1), True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-packages",
        default="/usr/local/lib/python3.12/dist-packages",
        help="Python site-packages directory containing sglang.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.site_packages)
    server_args = root / "sglang" / "srt" / "server_args.py"
    kvfp4_tensor = (
        root / "sglang" / "srt" / "layers" / "quantization" / "kvfp4_tensor.py"
    )

    server_text = server_args.read_text(encoding="utf-8")
    server_text, changed_fa4 = patch_once(
        server_text,
        '''                        KV4_FA4_MHA_BACKEND_CHOICES = [
                            "triton",
                            "torch_native",
                            "flex_attention",
                        ]
                        assert (
''',
        '''                        KV4_FA4_MHA_BACKEND_CHOICES = [
                            "triton",
                            "torch_native",
                            "flex_attention",
                        ]
                        if is_sm120_supported():
                            KV4_FA4_MHA_BACKEND_CHOICES.append("flashinfer")
                        assert (
''',
        "server_args FA4 MHA FlashInfer gate",
    )
    server_text, changed_mha = patch_once(
        server_text,
        '''                        KV4_ATTENTION_MHA_BACKEND_CHOICES = [
                            "triton",
                            "torch_native",
                            "flex_attention",
                            "trtllm_mha",
                        ]
                        assert (
''',
        '''                        KV4_ATTENTION_MHA_BACKEND_CHOICES = [
                            "triton",
                            "torch_native",
                            "flex_attention",
                            "trtllm_mha",
                        ]
                        if is_sm120_supported():
                            KV4_ATTENTION_MHA_BACKEND_CHOICES.append("flashinfer")
                        assert (
''',
        "server_args MHA FlashInfer gate",
    )

    kv_text = kvfp4_tensor.read_text(encoding="utf-8")
    kv_text, changed_alias = patch_once(
        kv_text,
        "\n\nclass NVFP4KVQuantizeUtil:\n",
        "\n\nKVFP4QuantizeUtil = BlockFP4KVQuantizeUtil\n\n\nclass NVFP4KVQuantizeUtil:\n",
        "KVFP4QuantizeUtil alias",
    )

    changes = {
        "server_args_fa4_gate": changed_fa4,
        "server_args_mha_gate": changed_mha,
        "kvfp4_alias": changed_alias,
    }
    print(changes)

    if not args.dry_run:
        server_args.write_text(server_text, encoding="utf-8")
        kvfp4_tensor.write_text(kv_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
