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


def patch_all(text: str, old: str, new: str, label: str) -> tuple[str, int]:
    if old not in text:
        return text, 0
    return text.replace(old, new), text.count(old)


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
    flashinfer_backend = (
        root / "sglang" / "srt" / "layers" / "attention" / "flashinfer_backend.py"
    )
    kvfp4_tensor = (
        root / "sglang" / "srt" / "layers" / "quantization" / "kvfp4_tensor.py"
    )
    memory_pool = root / "sglang" / "srt" / "mem_cache" / "memory_pool.py"

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

    pool_text = memory_pool.read_text(encoding="utf-8")
    pool_text, changed_fp4_raw = patch_once(
        pool_text,
        """    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
""",
        """    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def get_fp4_kv_buffer(self, layer_id: int):
        layer_idx = layer_id - self.start_layer
        return (
            self.k_buffer[layer_idx].view(self.dtype),
            self.v_buffer[layer_idx].view(self.dtype),
        )

    def get_fp4_kv_scale_buffer(self, layer_id: int):
        layer_idx = layer_id - self.start_layer
        return self.k_scale_buffer[layer_idx], self.v_scale_buffer[layer_idx]

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
""",
        "MHA FP4 raw KV accessor",
    )

    backend_text = flashinfer_backend.read_text(encoding="utf-8")
    backend_text, changed_backend_helper = patch_once(
        backend_text,
        """    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @debug_kernel_api
""",
        """    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _get_kv_buffer_for_flashinfer(self, token_to_kv_pool, layer_id: int):
        fp4_getter = getattr(token_to_kv_pool, "get_fp4_kv_buffer", None)
        if fp4_getter is not None:
            return fp4_getter(layer_id)
        return token_to_kv_pool.get_kv_buffer(layer_id)

    @debug_kernel_api
""",
        "FlashInfer packed FP4 KV helper",
    )
    backend_text, changed_backend_self_calls = patch_all(
        backend_text,
        "self.token_to_kv_pool.get_kv_buffer(layer.layer_id)",
        "self._get_kv_buffer_for_flashinfer(self.token_to_kv_pool, layer.layer_id)",
        "FlashInfer packed FP4 KV self call sites",
    )
    backend_text, changed_backend_batch_calls = patch_all(
        backend_text,
        "forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)",
        "self._get_kv_buffer_for_flashinfer(forward_batch.token_to_kv_pool, layer.layer_id)",
        "FlashInfer packed FP4 KV forward-batch call sites",
    )
    if changed_backend_self_calls + changed_backend_batch_calls == 0:
        raise RuntimeError("could not find FlashInfer packed FP4 KV call sites")

    changes = {
        "server_args_fa4_gate": changed_fa4,
        "server_args_mha_gate": changed_mha,
        "kvfp4_alias": changed_alias,
        "mha_fp4_raw_kv_accessor": changed_fp4_raw,
        "flashinfer_packed_fp4_kv_helper": changed_backend_helper,
        "flashinfer_packed_fp4_kv_self_call_sites": changed_backend_self_calls,
        "flashinfer_packed_fp4_kv_forward_batch_call_sites": changed_backend_batch_calls,
    }
    print(changes)

    if not args.dry_run:
        server_args.write_text(server_text, encoding="utf-8")
        kvfp4_tensor.write_text(kv_text, encoding="utf-8")
        memory_pool.write_text(pool_text, encoding="utf-8")
        flashinfer_backend.write_text(backend_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
