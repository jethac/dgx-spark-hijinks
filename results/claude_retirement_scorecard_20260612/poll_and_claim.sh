#!/usr/bin/env bash
# Window poller for the retirement scorecard block. Polls every 10 min.
# FREE = both marker paths absent AND docker ps empty.
# Claim after 2 consecutive FREE checks: write markers FIRST (noclobber),
# re-verify docker empty, else back off. Exit codes:
#   0 = CLAIMED, 2 = DEADLINE (>= 10:00 JST box time), 3 = STALL suspected
#       (marker present + docker empty for 3 consecutive checks).
set -u
HOST=jethac@100.113.98.11
LOG=${1:?usage: poll_and_claim.sh LOGFILE}
FREE_STREAK=0
STALL_STREAK=0

while true; do
  STATE=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" '
    m1=0; m2=0
    [ -e /home/jethac/CLAUDE_WINDOW_OPEN ] && m1=1
    [ -e /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN ] && m2=1
    d=$(docker ps -q | wc -l)
    h=$(date +%H%M)
    echo "$m1 $m2 $d $h"' 2>/dev/null)
  if [ -z "$STATE" ]; then
    echo "$(date -Is) SSH_FAIL" >> "$LOG"
    sleep 120
    continue
  fi
  read -r M1 M2 D HHMM <<< "$STATE"
  echo "$(date -Is) m1=$M1 m2=$M2 docker=$D boxtime=$HHMM free_streak=$FREE_STREAK stall_streak=$STALL_STREAK" >> "$LOG"

  # deadline: box local 10:00 JST (and not late-night wrap: deadline window 08:00-23:59 check)
  if [ "$HHMM" -ge 1000 ] && [ "$HHMM" -lt 2300 ]; then
    echo "$(date -Is) DEADLINE_REACHED" >> "$LOG"
    exit 2
  fi

  if [ "$M1" = "0" ] && [ "$M2" = "0" ] && [ "$D" = "0" ]; then
    FREE_STREAK=$((FREE_STREAK+1))
    STALL_STREAK=0
    if [ "$FREE_STREAK" -ge 2 ]; then
      CLAIM=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" '
        set -o noclobber
        ts=$(date -Is)
        ok=1
        { echo "claude retirement scorecard $ts" > /home/jethac/CLAUDE_WINDOW_OPEN; } 2>/dev/null || ok=0
        { echo "claude retirement scorecard $ts" > /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN; } 2>/dev/null || ok=0
        if [ "$ok" = "0" ]; then
          grep -ls "retirement scorecard" /home/jethac/CLAUDE_WINDOW_OPEN /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN 2>/dev/null | xargs -r rm -f
          echo CLAIM_LOST
          exit 0
        fi
        sleep 8
        if [ -n "$(docker ps -q)" ]; then
          rm -f /home/jethac/CLAUDE_WINDOW_OPEN /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN
          echo CLAIM_RACED
          exit 0
        fi
        echo CLAIM_OK' 2>/dev/null)
      echo "$(date -Is) CLAIM_RESULT=$CLAIM" >> "$LOG"
      if [ "$CLAIM" = "CLAIM_OK" ]; then
        exit 0
      fi
      FREE_STREAK=0
    fi
  elif [ "$D" = "0" ] && { [ "$M1" = "1" ] || [ "$M2" = "1" ]; }; then
    # marker held but no containers - possible stalled holder
    STALL_STREAK=$((STALL_STREAK+1))
    FREE_STREAK=0
    if [ "$STALL_STREAK" -ge 3 ]; then
      echo "$(date -Is) STALL_SUSPECTED" >> "$LOG"
      exit 3
    fi
  else
    FREE_STREAK=0
    STALL_STREAK=0
  fi
  sleep 600
done
