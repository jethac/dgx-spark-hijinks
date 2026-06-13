import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


MODEL = os.environ.get("MODEL", "google/gemma-4-E2B-it")
TOKEN = os.environ.get("HF_TOKEN")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/root/results/gemma4_chat_control"))
DTYPE = getattr(torch, os.environ.get("DTYPE", "float32"))
DEVICE = os.environ.get("DEVICE", "cuda")
RAW_PROMPT = os.environ.get("RAW_PROMPT", "The capital of France is")
CHAT_PROMPT = os.environ.get("CHAT_PROMPT", "What is the capital of France? Answer in one word.")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))


def decode_new(tok, generated, prompt_len):
    return tok.decode(generated[0, prompt_len:], skip_special_tokens=True)


def run_case(model, tok, label, inputs):
    if DEVICE == "cuda":
        inputs = inputs.to("cuda")
    with torch.inference_mode():
        out = model(**inputs)
        logits = out.logits[:, -1, :].detach().float()
        top = torch.topk(logits, k=8, dim=-1)
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return {
        "label": label,
        "prompt_len": int(inputs.input_ids.shape[-1]),
        "generated": decode_new(tok, generated.detach().cpu(), inputs.input_ids.shape[-1]),
        "top_tokens": [
            {
                "token_id": int(tid),
                "token": tok.decode([int(tid)]),
                "logit": float(logit),
            }
            for tid, logit in zip(top.indices[0].cpu(), top.values[0].cpu())
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        token=TOKEN,
        dtype=DTYPE,
        device_map={"": DEVICE} if DEVICE == "cuda" else None,
    ).eval()
    if DEVICE == "cpu":
        model.to("cpu")

    raw_inputs = tok(RAW_PROMPT, return_tensors="pt")
    chat_inputs = tok.apply_chat_template(
        [{"role": "user", "content": CHAT_PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    payload = {
        "meta": {
            "model": MODEL,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
            "dtype": str(DTYPE),
            "run_device": DEVICE,
            "raw_prompt": RAW_PROMPT,
            "chat_prompt": CHAT_PROMPT,
        },
        "raw": run_case(model, tok, "raw", raw_inputs),
        "chat": run_case(model, tok, "chat_template", chat_inputs),
    }
    out_path = OUT_DIR / "chat_control.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RAW_GEN", repr(payload["raw"]["generated"]), flush=True)
    print("CHAT_GEN", repr(payload["chat"]["generated"]), flush=True)
    print("OUT", out_path, flush=True)


if __name__ == "__main__":
    main()
