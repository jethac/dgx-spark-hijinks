#!/usr/bin/env python3
"""Multimodal AUDIO-needle eval for KV-cache quality.

The text and image needles never touch the AUDIO KV path (Gemma 4 / Gemma-3n lineage carries a USM
conformer audio encoder; audio spans get their own KV tokens and are special-cased in masking). This
plants a 6-digit access code SPOKEN in one audio clip, hides it among N filler clips, and asks the
model to read it back. That exercises the quantized audio-token KV. A KV dtype that holds audio
retrieval like bf16/fp8 is fine; one that drops it has a real audio-KV defect. Same seed per config.

Audio is synthesized with gTTS (natural speech — robotic espeak risks the bf16 smoke test itself).
bf16 is the smoke test: if bf16 can't transcribe the needle, the harness/audio-support is the problem,
not the KV dtype. Only compare fp8/nvfp4 once bf16 passes.
"""
import argparse
import base64
import io
import json
import os
import random
import librosa
import soundfile as sf
from gtts import gTTS
from vllm import LLM, SamplingParams


def _spoken(text, rng):
    """gTTS mp3 -> 16kHz PCM WAV bytes. vLLM's load_audio tries soundfile first, which reads PCM
    WAV natively (no pyav needed); soundfile can't decode mp3, so we transcode via librosa here."""
    mp3 = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(mp3)
    mp3.seek(0)
    y, sr = librosa.load(mp3, sr=16000, mono=True)
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _spell(code):
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    return " ".join(digits[d] for d in code)


def audio_uri(wav_bytes):
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()


def build(trials, n_audio, seed=0):
    rng = random.Random(seed)
    items = []
    for _ in range(trials):
        code = str(rng.randint(100000, 999999))
        pos = rng.randint(0, n_audio - 1)
        clips = []
        for i in range(n_audio):
            if i == pos:
                txt = f"The access code is {_spell(code)}."
            else:
                txt = f"Item number {rng.randint(10, 99)} is in the warehouse."
            clips.append(_spoken(txt, rng))
        content = [{"type": "audio_url", "audio_url": {"url": audio_uri(w)}} for w in clips]
        content.append({"type": "text", "text": (
            "Exactly one of the audio clips speaks an access code (a 6-digit number). "
            "What is that 6-digit access code? Answer with only the digits, no spaces.")})
        items.append({"code": code, "pos": pos,
                      "messages": [{"role": "user", "content": content}]})
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--kv-cache-dtype", required=True)
    p.add_argument("--n-audio", type=int, default=6)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    items = build(args.trials, args.n_audio)
    llm = LLM(model=args.model, kv_cache_dtype=args.kv_cache_dtype,
              max_model_len=args.max_model_len, enforce_eager=True, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"audio": args.n_audio}, enable_prefix_caching=False)
    outs = llm.chat([it["messages"] for it in items],
                    SamplingParams(temperature=0.0, max_tokens=16))
    n_ok = 0
    for it, o in zip(items, outs):
        gen = o.outputs[0].text.replace(" ", "")
        n_ok += int(it["code"] in gen)
    report = {"label": args.label, "kv_cache_dtype": args.kv_cache_dtype,
              "n_audio": args.n_audio, "n_queries": len(items),
              "audio_needle_accuracy": round(n_ok / len(items), 3)}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
