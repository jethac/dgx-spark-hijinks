#!/usr/bin/env python3
"""Resolve PURELY-ADDITIVE cherry-pick conflicts: for each <<<<<<< HEAD / ======= / >>>>>>> hunk,
if the HEAD (ours/e3) side is empty/whitespace, the commit is only ADDING code where upstream has
nothing -> keep theirs (the added block) and drop the markers. Non-additive hunks (HEAD side has
real content) are LEFT intact and reported, so a human resolves those. Prints per-file: resolved N,
left M. Usage: resolve_additive.py <file> [<file> ...]"""
import sys

def resolve(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    out, i, resolved, left = [], 0, 0, 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            j = i + 1
            head = []
            while j < len(lines) and not lines[j].startswith("======="):
                head.append(lines[j]); j += 1
            k = j + 1
            theirs = []
            while k < len(lines) and not lines[k].startswith(">>>>>>>"):
                theirs.append(lines[k]); k += 1
            # k is the >>>>>>> line
            if "".join(head).strip() == "":          # additive: HEAD side empty
                out.extend(theirs); resolved += 1
            else:                                      # real conflict: keep markers
                out.extend(lines[i:k + 1]); left += 1
            i = k + 1
        else:
            out.append(lines[i]); i += 1
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"  {path}: resolved {resolved} additive, left {left} real")
    return left

total_left = 0
for p in sys.argv[1:]:
    total_left += resolve(p)
sys.exit(1 if total_left else 0)
