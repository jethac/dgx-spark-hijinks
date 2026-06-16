#!/usr/bin/env python3
"""Multimodal image-needle eval for KV-cache quality.

The text needle never touches the vision KV. This plants a 6-digit code rendered INTO one image,
hides it among N filler images, and asks the model to read it back. That exercises the quantized
image-token KV (each Gemma image is ~256 KV tokens). A KV dtype that holds image retrieval like
bf16/fp8 is fine; one that drops it has a real vision-KV defect. Run once per KV config, same seed.
"""
import argparse
import base64
import io
import json
import random

from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def filler_img(rng):
    img = Image.new("RGB", (448, 448), (rng.randint(40, 220),) * 3)
    d = ImageDraw.Draw(img)
    d.text((40, 200), f"item number {rng.randint(10, 99)}", fill=(255, 255, 255), font=_font(36))
    return img


def needle_img(code):
    img = Image.new("RGB", (448, 448), (15, 15, 40))
    d = ImageDraw.Draw(img)
    d.text((30, 150), "ACCESS CODE", fill=(255, 230, 0), font=_font(48))
    d.text((60, 230), code, fill=(255, 230, 0), font=_font(72))
    return img


def img_uri(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build(trials, n_images, seed=0):
    rng = random.Random(seed)
    items = []
    for _ in range(trials):
        code = str(rng.randint(100000, 999999))
        pos = rng.randint(0, n_images - 1)
        imgs = [filler_img(rng) for _ in range(n_images)]
        imgs[pos] = needle_img(code)
        content = [{"type": "image_url", "image_url": {"url": img_uri(im)}} for im in imgs]
        content.append({"type": "text", "text": (
            "Exactly one of the images shows an ACCESS CODE (a 6-digit number). "
            "What is that 6-digit access code? Answer with only the number.")})
        items.append({"code": code, "pos": pos,
                      "messages": [{"role": "user", "content": content}]})
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--kv-cache-dtype", required=True)
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    items = build(args.trials, args.n_images)
    llm = LLM(model=args.model, kv_cache_dtype=args.kv_cache_dtype,
              max_model_len=args.max_model_len, enforce_eager=True, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": args.n_images}, enable_prefix_caching=False)
    outs = llm.chat([it["messages"] for it in items],
                    SamplingParams(temperature=0.0, max_tokens=16))
    n_ok = 0
    for it, o in zip(items, outs):
        gen = o.outputs[0].text
        n_ok += int(it["code"] in gen)
    report = {"label": args.label, "kv_cache_dtype": args.kv_cache_dtype,
              "n_images": args.n_images, "n_queries": len(items),
              "image_needle_accuracy": round(n_ok / len(items), 3)}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
