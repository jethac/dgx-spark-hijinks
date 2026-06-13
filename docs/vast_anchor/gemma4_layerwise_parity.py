import gc
import json
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


MODEL = os.environ.get("MODEL", "google/gemma-4-12B-it")
PROMPT = os.environ.get("PROMPT", "The capital of France is")
TOKEN = os.environ.get("HF_TOKEN")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/root/results/gemma4_layerwise_parity"))
MAX_HOOKS = int(os.environ.get("MAX_HOOKS", "120"))
MAX_LAYERS = int(os.environ.get("MAX_LAYERS", "4"))


def summarize_tensor(t):
    if isinstance(t, (tuple, list)):
        for item in t:
            if torch.is_tensor(item):
                return summarize_tensor(item)
        return None
    if not torch.is_tensor(t):
        return None
    x = t.detach()
    if not x.is_floating_point():
        x = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "mean": float(x.float().mean().cpu()),
        "std": float(x.float().std(unbiased=False).cpu()),
        "amax": float(x.float().abs().max().cpu()),
        "nan": int(torch.isnan(x.float()).sum().cpu()),
        "inf": int(torch.isinf(x.float()).sum().cpu()),
    }


def first_tensor(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def tensor_compare(a, b):
    if a is None or b is None:
        return None
    if list(a.shape) != list(b.shape):
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    af = a.float().reshape(-1).cpu()
    bf = b.float().reshape(-1).cpu()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cosine = float(torch.dot(af, bf) / denom) if float(denom) != 0.0 else math.nan
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine": cosine,
        "amax_a": float(af.abs().max()),
        "amax_b": float(bf.abs().max()),
    }


def candidate_modules(model):
    picked = OrderedDict()
    names = dict(model.named_modules())
    for name, module in names.items():
        lname = name.lower()
        if not name:
            continue
        if any(key in lname for key in ("embed", "norm", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")):
            picked[name] = module
        if len(picked) >= MAX_HOOKS:
            break
    filtered = OrderedDict()
    for name, module in picked.items():
        layer_idx = None
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part in {"layers", "layer"} and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None or layer_idx < MAX_LAYERS:
            filtered[name] = module
    return filtered


@contextmanager
def hook_outputs(model):
    records = OrderedDict()
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            if name not in records:
                records[name] = first_tensor(output)
        return hook

    for name, module in candidate_modules(model).items():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def load_model(dtype):
    return AutoModelForImageTextToText.from_pretrained(
        MODEL,
        token=TOKEN,
        dtype=dtype,
        device_map={"": "cuda"},
    ).eval()


def run_once(dtype_name, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
    model = load_model(dtype)
    inputs = tok(PROMPT, return_tensors="pt").to("cuda")
    with torch.inference_mode(), hook_outputs(model) as records:
        out = model(**inputs)
        logits = out.logits[:, -1, :].detach()
        top = torch.topk(logits.float(), k=8, dim=-1)
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_text = tok.decode(generated[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    result = {
        "dtype": dtype_name,
        "prompt": PROMPT,
        "generated": gen_text,
        "top_tokens": [
            {
                "token_id": int(tid),
                "token": tok.decode([int(tid)]),
                "logit": float(logit),
            }
            for tid, logit in zip(top.indices[0].cpu(), top.values[0].cpu())
        ],
        "logits_summary": summarize_tensor(logits),
        "records": {name: summarize_tensor(t) for name, t in records.items()},
    }
    tensors = {name: t.cpu() for name, t in records.items()}
    logits_cpu = logits.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result, tensors, logits_cpu


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": MODEL,
        "prompt": PROMPT,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "max_layers": MAX_LAYERS,
        "max_hooks": MAX_HOOKS,
    }
    print("META", json.dumps(meta, sort_keys=True), flush=True)

    fp32, fp32_tensors, fp32_logits = run_once("float32", torch.float32)
    bf16, bf16_tensors, bf16_logits = run_once("bfloat16", torch.bfloat16)

    common = [name for name in fp32_tensors if name in bf16_tensors]
    comparisons = OrderedDict()
    for name in common:
        comparisons[name] = tensor_compare(fp32_tensors[name], bf16_tensors[name])
    comparisons["__final_logits__"] = tensor_compare(fp32_logits, bf16_logits)

    first_bad = None
    for name, comp in comparisons.items():
        if not comp or "cosine" not in comp:
            continue
        if comp["cosine"] < 0.99 or comp["max_abs"] > 5.0:
            first_bad = {"name": name, **comp}
            break

    payload = {
        "meta": meta,
        "fp32": fp32,
        "bf16": bf16,
        "comparisons": comparisons,
        "first_bad": first_bad,
    }
    (OUT_DIR / "layerwise_parity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("FP32_GEN", repr(fp32["generated"]), flush=True)
    print("BF16_GEN", repr(bf16["generated"]), flush=True)
    print("FIRST_BAD", json.dumps(first_bad, sort_keys=True), flush=True)
    print("OUT", OUT_DIR / "layerwise_parity.json", flush=True)


if __name__ == "__main__":
    main()
