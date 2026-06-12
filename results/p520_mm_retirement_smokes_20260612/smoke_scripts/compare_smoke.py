#!/usr/bin/env python3
"""Adjudicate two banked smoke JSONs for the mm-retire gates.

Gate (b) FI-route vs Triton-route SEMANTIC equivalence (image smoke):
  both must be image-grounded (same keyword set), and the first-reply
  texts compared. Image mm spans are bidirectional-masked differently
  per route only if the masking is wrong; a correct mm path yields the
  SAME grounded answer. We report exact-match plus a token-overlap
  fallback (semantic, not necessarily byte-identical across two distinct
  attention kernels).

Gate (c) text-only token-identity (knob-on vs knob-off, SAME backend):
  here we DEMAND byte-identical content AND identical token seqs (the mm
  knob must not perturb pure-text serving on the same backend).

Mode is chosen by --mode {semantic,identical}.
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def first_content(d: dict):
    cs = d.get("contents") or []
    return cs[0] if cs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--mode", choices=["semantic", "identical"], required=True)
    ap.add_argument("--label", default="compare")
    ap.add_argument("--output")
    args = ap.parse_args()

    A = load(args.a)
    B = load(args.b)
    ca = first_content(A)
    cb = first_content(B)

    report = {
        "schema": "smoke-compare/v1",
        "label": args.label,
        "mode": args.mode,
        "a_file": args.a,
        "b_file": args.b,
        "a_content": ca,
        "b_content": cb,
        "a_ok": A.get("ok"),
        "b_ok": B.get("ok"),
    }

    both_ran = isinstance(ca, str) and isinstance(cb, str) and ca.strip() and cb.strip()

    if args.mode == "identical":
        # byte-identical content AND identical token seqs
        ta = (A.get("token_seqs") or [None])[0]
        tb = (B.get("token_seqs") or [None])[0]
        content_id = both_ran and (ca == cb)
        tok_id = ta == tb
        report["content_identical"] = bool(content_id)
        report["tokens_identical"] = bool(tok_id)
        report["verdict"] = (
            "GREEN: token-identical"
            if (content_id and tok_id)
            else "RED: NOT token-identical"
        )
        report["ok"] = bool(content_id and tok_id)
    else:
        # semantic: exact match is best; else report word overlap
        exact = both_ran and (ca == cb)
        wa = set((ca or "").lower().split())
        wb = set((cb or "").lower().split())
        overlap = (len(wa & wb) / max(1, len(wa | wb))) if (wa or wb) else 0.0
        report["exact_match"] = bool(exact)
        report["word_jaccard"] = round(overlap, 4)
        report["both_grounded"] = bool(A.get("ok_grounded") and B.get("ok_grounded"))
        # Semantic equivalence: both grounded AND (exact OR high overlap)
        sem_ok = report["both_grounded"] and (exact or overlap >= 0.6)
        report["verdict"] = (
            "GREEN: semantically equivalent"
            if sem_ok
            else "RED: routes diverge (inspect verbatim)"
        )
        report["ok"] = bool(sem_ok)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
