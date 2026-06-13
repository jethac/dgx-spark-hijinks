import gc
import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


MODEL = os.environ.get("MODEL", "google/gemma-4-E2B-it")
PROMPT = os.environ.get("PROMPT", "The capital of France is")
TOKEN = os.environ.get("HF_TOKEN")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/root/results/gemma4_cpu_gpu_localize"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8"))
MAX_HOOKS = int(os.environ.get("MAX_HOOKS", "320"))
MAX_LAYERS = int(os.environ.get("MAX_LAYERS", "-1"))
BAD_COSINE = float(os.environ.get("BAD_COSINE", "0.99"))
BAD_MAX_ABS = float(os.environ.get("BAD_MAX_ABS", "5.0"))


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


def summarize_tensor(t):
    if t is None:
        return None
    x = t.detach()
    if not x.is_floating_point():
        x = x.float()
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "mean": float(xf.mean().cpu()),
        "std": float(xf.std(unbiased=False).cpu()),
        "amax": float(xf.abs().max().cpu()),
        "nan": int(torch.isnan(xf).sum().cpu()),
        "inf": int(torch.isinf(xf).sum().cpu()),
    }


def tensor_compare(a, b):
    if a is None or b is None:
        return None
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    if af.numel() != bf.numel():
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "error": "shape_mismatch"}
    diff = (af - bf).abs()
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    cosine = float((torch.dot(af, bf) / denom).cpu()) if float(denom.cpu()) != 0.0 else None
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "cosine": cosine,
        "amax_a": float(af.abs().max().cpu()),
        "amax_b": float(bf.abs().max().cpu()),
    }


def layer_index(name):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in {"layers", "layer"} and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def candidate_modules(model):
    wanted_suffixes = (
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "v_norm",
        "gate_proj",
        "up_proj",
        "down_proj",
        "router",
        "gate",
        "lm_head",
    )
    out = OrderedDict()
    for name, module in model.named_modules():
        if len(out) >= MAX_HOOKS:
            break
        if not any(name.endswith(suffix) for suffix in wanted_suffixes):
            continue
        idx = layer_index(name)
        if idx is not None and MAX_LAYERS >= 0 and idx >= MAX_LAYERS:
            continue
        out[name] = module
    return out


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


def load_model(device):
    kwargs = {
        "token": TOKEN,
        "dtype": torch.float32,
    }
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda"}
    model = AutoModelForImageTextToText.from_pretrained(MODEL, **kwargs).eval()
    if device == "cpu":
        model.to("cpu")
    return model


def run_once(label, device):
    tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
    model = load_model(device)
    inputs = tok(PROMPT, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to("cuda")
    with torch.inference_mode(), hook_outputs(model) as records:
        out = model(**inputs)
        logits = out.logits[:, -1, :].detach()
        top = torch.topk(logits.float(), k=8, dim=-1)
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen_text = tok.decode(generated[0, inputs.input_ids.shape[-1]:].detach().cpu(), skip_special_tokens=True)
    result = {
        "label": label,
        "device": device,
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
    tensors = {name: t.detach().cpu() for name, t in records.items() if t is not None}
    logits_cpu = logits.detach().cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, tensors, logits_cpu


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": MODEL,
        "prompt": PROMPT,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        "max_layers": MAX_LAYERS,
        "max_hooks": MAX_HOOKS,
    }
    print("META", json.dumps(meta, sort_keys=True), flush=True)

    cpu, cpu_tensors, cpu_logits = run_once("cpu_fp32", "cpu")
    print("CPU_GEN", repr(cpu["generated"]), flush=True)
    gpu, gpu_tensors, gpu_logits = run_once("gpu_fp32", "cuda")
    print("GPU_GEN", repr(gpu["generated"]), flush=True)

    comparisons = OrderedDict()
    for name in cpu_tensors:
        if name in gpu_tensors:
            comparisons[name] = tensor_compare(cpu_tensors[name], gpu_tensors[name])
    comparisons["__final_logits__"] = tensor_compare(cpu_logits, gpu_logits)

    first_bad = None
    for name, comp in comparisons.items():
        if not comp or "cosine" not in comp or comp["cosine"] is None:
            continue
        if comp["cosine"] < BAD_COSINE or comp["max_abs"] > BAD_MAX_ABS:
            first_bad = {"name": name, **comp}
            break

    payload = {
        "meta": meta,
        "cpu": cpu,
        "gpu": gpu,
        "comparisons": comparisons,
        "first_bad": first_bad,
    }
    out_path = OUT_DIR / "cpu_gpu_localize.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("FIRST_BAD", json.dumps(first_bad, sort_keys=True), flush=True)
    print("OUT", out_path, flush=True)


if __name__ == "__main__":
    main()
