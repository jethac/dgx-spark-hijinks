#!/usr/bin/env bash
# 4-box parallel vLLM Gemma-4 anchor ladder on vast.ai.
# Usage:  VAST_API_KEY=... HF_TOKEN=... ./run_parallel_ladder.sh <CLEAN_IMAGE_REF>
# Secrets come ONLY from env (never written to a file / committed). Boxes are destroyed
# the instant their JSON is banked, plus an unconditional cleanup sweep at the end.
set -uo pipefail
IMAGE="${1:?need clean x86 sm_120 image ref (ghcr...)}"
: "${VAST_API_KEY:?set VAST_API_KEY in env}" ; : "${HF_TOKEN:?set HF_TOKEN in env}"
VAST="${VAST:-vastai}" ; HARNESS="$(dirname "$0")/eval_harness.py"
GATE="$(dirname "$0")/gen_test.py" ; KEY="$HOME/.ssh/id_ed25519"
OUT="results/vast_ladder_$(date -u +%Y%m%dT%H%M%SZ)" ; mkdir -p "$OUT"
# TODO(finalize on real image): PYBIN + PYENV are how the harness runs INSIDE the image's
# container. If Codex bakes a venv at /opt/venv with flashinfer importable, set accordingly.
PYBIN="${PYBIN:-python3}" ; PYENV="${PYENV:-}"   # e.g. PYENV='PYTHONPATH=/opt/flashinfer'

pick() {  # pick(gpu_substr, min_gpu_ram_mib) -> cheapest reliable offer id
  $VAST search offers "num_gpus=1 rentable=true cuda_max_good>=13.0" -o dph_total --raw \
   | $PYBIN -c "import sys,json;g=sys.argv[1];m=int(sys.argv[2]);d=json.load(sys.stdin);\
o=[x for x in d if g in x['gpu_name'] and x['gpu_ram']>=m and x['reliability2']>0.97 and x['inet_down']>500 and x['disk_space']>120];\
o.sort(key=lambda x:x['dph_total']);print(o[0]['id'] if o else '')" "$1" "$2"; }

boot() {  # boot(offer_id) -> "iid host port"  (waits for ssh)
  local iid host port; iid=$($VAST create instance "$1" --image "$IMAGE" --disk 160 --ssh --direct --raw \
        | $PYBIN -c "import sys,json;print(json.load(sys.stdin)['new_contract'])")
  $VAST attach ssh "$iid" "$(cat "$KEY.pub")" >/dev/null 2>&1
  for _ in $(seq 1 40); do
    read host port < <($VAST ssh-url "$iid" 2>/dev/null | sed -E 's#ssh://root@##;s#:# #')
    [ -n "${host:-}" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p "$port" "root@$host" true 2>/dev/null && { echo "$iid $host $port"; return; }
    sleep 15
  done; echo "$iid  "; }

run_rung() {  # run_rung(label model gpu ram "envextra" "harness_args...") -> backgrounded
  local label="$1" model="$2" gpu="$3" ram="$4" envx="$5"; shift 5
  ( local off iid host port; off=$(pick "$gpu" "$ram"); [ -z "$off" ] && { echo "$label: NO OFFER" >"$OUT/$label.err"; return; }
    read iid host port < <(boot "$off"); echo "$label box=$iid $host:$port off=$off" >"$OUT/$label.box"
    [ -z "$host" ] && { echo "$label: ssh never came up ($iid)" >"$OUT/$label.err"; $VAST destroy instance "$iid" <<<y >/dev/null 2>&1; return; }
    scp -o StrictHostKeyChecking=no -P "$port" "$HARNESS" "root@$host:/root/eval_harness.py" >/dev/null 2>&1
    for kv in bfloat16 nvfp4; do
      ssh -o StrictHostKeyChecking=no -p "$port" "root@$host" \
        "export HF_TOKEN='$HF_TOKEN' ANCHOR_MODEL='$model' $envx $PYENV; cd /root && $PYBIN eval_harness.py $kv /root/out_${kv}.json $* >/root/${kv}.log 2>&1; echo rc=\$?" >>"$OUT/$label.run" 2>&1
      scp -o StrictHostKeyChecking=no -P "$port" "root@$host:/root/out_${kv}.json" "$OUT/${label}_${kv}.json" >/dev/null 2>&1
    done
    $VAST destroy instance "$iid" <<<y >/dev/null 2>&1; echo "$label DONE, box destroyed" >>"$OUT/$label.box"
  ) & }

echo "== GREEN GATE on 1x5090 =="
goff=$(pick "RTX 5090" 30000); read giid ghost gport < <(boot "$goff")
scp -o StrictHostKeyChecking=no -P "$gport" "$GATE" "root@$ghost:/root/gen_test.py" >/dev/null 2>&1
ssh -o StrictHostKeyChecking=no -p "$gport" "root@$ghost" "export HF_TOKEN='$HF_TOKEN' $PYENV; cd /root && $PYBIN gen_test.py 2>/dev/null" | tee "$OUT/green_gate.log"
$VAST destroy instance "$giid" <<<y >/dev/null 2>&1
grep -qi "Paris" "$OUT/green_gate.log" || { echo "GREEN GATE FAILED — not fanning out. See $OUT/green_gate.log"; exit 1; }

echo "== FAN OUT 4 BOXES =="
run_rung box1_12b      google/gemma-4-12B-it "RTX 5090"        30000 "" --ctx 8185 --prefix 4096
run_rung box2_12bsweep google/gemma-4-12B-it "RTX 5090"        30000 "ANCHOR_SWEEP=0,1024,2048,4096"
run_rung box3_31b      google/gemma-4-31B-it "RTX PRO 6000"    90000 "VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1" --ctx 8185 --prefix 4096
run_rung box4_e4b      google/gemma-4-E4B-it "RTX PRO 6000"    90000 "" --ctx 8185 --prefix 4096
wait
echo "== CLEANUP SWEEP (destroy any straggler) =="
$VAST show instances --raw | $PYBIN -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin)]" | while read i; do $VAST destroy instance "$i" <<<y >/dev/null 2>&1; done
echo "results in $OUT"; ls -la "$OUT"
