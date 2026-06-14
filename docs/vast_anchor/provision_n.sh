#!/usr/bin/env bash
# Usage: provision_n.sh <slot>  -> provisions one box, writes ~/pfx_box<slot>.txt + ~/pfx_iid<slot>.txt
export VAST_API_KEY=09307f0e70a2601b026ea7563986376c54b6237f12c3b4347225a650dbe786ed
V=/home/jetha/vllm_wheel_env/bin/vastai; PUB="$(cat ~/.ssh/id_ed25519.pub)"; S="$1"
echo "" > ~/pfx_box$S.txt; echo "" > ~/pfx_iid$S.txt
OFF=$($V search offers "num_gpus=1 rentable=true cuda_max_good>=13.0 verified=true" -o dph_total --raw 2>/dev/null | /home/jetha/vllm_wheel_env/bin/python -c "import sys,json,random;d=json.load(sys.stdin);o=[x for x in d if 'PRO 6000' in x['gpu_name'] and x['reliability2']>0.99 and x['disk_space']>140 and x['machine_id']!=51732];o.sort(key=lambda x:x['dph_total']);print(o[($S-1)%len(o)]['id']) if o else print('NONE')")
IID=$($V create instance $OFF --image nvidia/cuda:13.0.1-devel-ubuntu24.04 --disk 170 --ssh --raw 2>&1 | /home/jetha/vllm_wheel_env/bin/python -c "import sys,json;print(json.load(sys.stdin)['new_contract'])")
echo $IID > ~/pfx_iid$S.txt; echo "[slot $S] iid=$IID off=$OFF"
sleep 12; $V attach ssh $IID "$PUB" >/dev/null 2>&1
for i in $(seq 1 44); do
  hp=$($V ssh-url $IID 2>/dev/null | sed -E "s#ssh://root@##"); host=${hp%%:*}; port=${hp##*:}
  if [ -n "$host" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p "$port" "root@$host" "echo OK" 2>/dev/null | grep -q OK; then echo "[slot $S] READY $host:$port"; echo "$host $port" > ~/pfx_box$S.txt; break; fi; sleep 13
done
read H P < ~/pfx_box$S.txt
[ -z "$H" ] && { echo "[slot $S] FAILED"; exit 1; }
R=/mnt/b/workshop/worktrees/dgx-spark-hijinks/spark-hijinks-022-gemma4-mixed-kv/docs/vast_anchor
scp -o StrictHostKeyChecking=no -P $P "$R/e2e_setup_wget.sh" "$R/vllm_matched_kv_anchor.py" "$R/corpus_fetch.py" root@$H:/root/ >/dev/null 2>&1
ssh -o StrictHostKeyChecking=no -p $P root@$H "cd /root && sed -i 's/\r\$//' *.sh *.py && nohup bash e2e_setup_wget.sh > setup.log 2>&1 </dev/null & sleep 2; echo [slot $S] SETUP_LAUNCHED"
