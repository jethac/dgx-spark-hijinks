# SPDX-License-Identifier: Apache-2.0
"""Deterministic audio assets for the Gemma 4 audio mm retirement cells.

Amendment 5 (OVERNIGHT_LADDER_PLAN_20260612): two clips, both 16 kHz
mono WAV, <= 15 s, banked with the results:

1. speech_librispeech_1272-128104-0000.wav - real speech with a KNOWN
   transcript, taken from the LibriSpeech dev-clean corpus via the HF
   dataset sample mirror hf-internal-testing/librispeech_asr_dummy
   (config "clean", split "validation", row 0; sample id
   1272-128104-0000, speaker 1272, chapter 128104).
   The reference transcript is read from the dataset itself and written
   to speech_transcript.txt verbatim.

2. tone_control.wav - synthesized content-free control: 1 s of 440 Hz,
   1 s of silence, 1 s of 880 Hz, 1 s of silence, amplitude 0.3,
   generated from exact integer-sample sinusoids (bit-deterministic,
   no RNG).

Usage: python make_audio_mm_assets.py <out_dir>
Writes WAVs + speech_transcript.txt + assets_manifest.json (md5s,
provenance) into <out_dir>.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def make_speech(out_dir: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    row = ds[0]
    assert row["id"] == "1272-128104-0000", row["id"]
    audio = row["audio"]
    wav = np.asarray(audio["array"], dtype=np.float32)
    assert audio["sampling_rate"] == SR, audio["sampling_rate"]
    dur = len(wav) / SR
    assert dur <= 15.0, dur
    out = out_dir / f"speech_librispeech_{row['id']}.wav"
    sf.write(out, wav, SR, subtype="PCM_16")
    transcript = row["text"]
    (out_dir / "speech_transcript.txt").write_text(transcript + "\n")
    return {
        "file": out.name,
        "source": (
            "LibriSpeech ASR corpus (Panayotov et al., 2015, CC BY 4.0), "
            "dev-clean utterance 1272-128104-0000, via HF dataset "
            "hf-internal-testing/librispeech_asr_dummy config=clean "
            "split=validation row=0"
        ),
        "speaker": "1272",
        "chapter": "128104",
        "sampling_rate": SR,
        "duration_s": round(dur, 3),
        "reference_transcript": transcript,
        "md5": md5(out),
    }


def make_tone_control(out_dir: Path) -> dict:
    t1 = np.arange(SR, dtype=np.float64) / SR
    seg_440 = 0.3 * np.sin(2 * np.pi * 440.0 * t1)
    seg_880 = 0.3 * np.sin(2 * np.pi * 880.0 * t1)
    silence = np.zeros(SR, dtype=np.float64)
    wav = np.concatenate([seg_440, silence, seg_880, silence]).astype(np.float32)
    out = out_dir / "tone_control.wav"
    sf.write(out, wav, SR, subtype="PCM_16")
    return {
        "file": out.name,
        "source": "synthesized by make_audio_mm_assets.py (no RNG)",
        "pattern": "1s 440Hz sine, 1s silence, 1s 880Hz sine, 1s silence",
        "amplitude": 0.3,
        "sampling_rate": SR,
        "duration_s": 4.0,
        "md5": md5(out),
    }


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "speech": make_speech(out_dir),
        "tone_control": make_tone_control(out_dir),
    }
    mpath = out_dir / "assets_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
