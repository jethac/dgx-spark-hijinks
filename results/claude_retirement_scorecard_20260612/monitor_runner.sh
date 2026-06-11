#!/usr/bin/env bash
set -u
HOST=jethac@100.113.98.11
LOG=${1:?}
while true; do
  OUT=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" '
    S=/home/jethac/spark_tmp/claude_retirement_scorecard_20260612/status.txt
    if grep -q SCORECARD_DONE "$S" 2>/dev/null; then echo DONE
    elif kill -0 1903469 2>/dev/null; then echo "RUNNING $(tail -1 "$S" 2>/dev/null)"
    else echo "DEAD $(tail -1 "$S" 2>/dev/null)"; fi' 2>/dev/null)
  echo "$(date -Is) ${OUT:-SSH_FAIL}" >> "$LOG"
  case "$OUT" in
    DONE*) exit 0 ;;
    DEAD*) exit 4 ;;
  esac
  sleep 600
done
