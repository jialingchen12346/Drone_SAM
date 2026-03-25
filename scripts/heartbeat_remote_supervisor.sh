#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/research_workspace/artifacts/heartbeat"
PID_FILE="${STATE_DIR}/heartbeat.pid"
LOG_FILE="${STATE_DIR}/heartbeat.log"
ALERT_FILE="${STATE_DIR}/alerts.log"
STATUS_FILE="${STATE_DIR}/latest_status.txt"

REMOTE_HOST="connect.bjb1.seetacloud.com"
REMOTE_PORT="37143"
REMOTE_USER="root"
REMOTE_REPO="/root/Drone-SAM-Adapter"
SSH_KEY="${HOME}/.ssh/id_ed25519_seeta_heartbeat"

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
TARGET_FREE_GB="${TARGET_FREE_GB:-12}"
STALE_SEC="${STALE_SEC:-600}"
AUTO_RECOVER="${AUTO_RECOVER:-1}"
AUTO_DISK_CLEANUP="${AUTO_DISK_CLEANUP:-1}"
KEEP_TOP_N_CHECKPOINTS="${KEEP_TOP_N_CHECKPOINTS:-1}"

REMOTE_TRAIN_LOG="/tmp/train_rrf_dsd_gen30_nccl_20260324.log"
REMOTE_SCHED_LOG="/tmp/auto_loop_scheduler_20260324.log"

mkdir -p "${STATE_DIR}"

ts() {
  date '+%F %T'
}

log() {
  local msg="$*"
  local line="[$(ts)] ${msg}"
  echo "${line}" >> "${LOG_FILE}"
  if [[ -t 1 ]]; then
    echo "${line}"
  fi
}

alert() {
  local msg="$*"
  local line="[$(ts)] ${msg}"
  echo "${line}" >> "${ALERT_FILE}"
  echo "${line}" >> "${LOG_FILE}"
  if [[ -t 1 ]]; then
    echo "${line}"
  fi
}

ssh_run() {
  ssh \
    -i "${SSH_KEY}" \
    -o BatchMode=yes \
    -o ConnectTimeout=12 \
    -o StrictHostKeyChecking=accept-new \
    -p "${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

collect_remote_status() {
  local cmd
  cmd=$(cat <<'EOF'
set -euo pipefail
train_cnt=$(ps -eo pid=,cmd= | awk '/segmentation\/train_cacaf.py/ && $0 !~ /heartbeat_remote_supervisor/ {c++} END{print c+0}')
loop_cnt=$(ps -eo pid=,cmd= | awk '(/scripts\/auto_train_eval_loop.py/ || /auto_loop_scheduler_20260324\.log/) && $0 !~ /heartbeat_remote_supervisor/ {c++} END{print c+0}')
df_avail_gb=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, - || echo "NA")
active_work_dir=$(ps -eo cmd= | awk '/segmentation\/train_cacaf.py/ {
  for (i=1; i<=NF; i++) {
    if ($i=="--work-dir") {
      print $(i+1)
    }
  }
}' | tail -n 1)
train_mtime=0
train_tail=""
latest_auto_train_log=$(ls -1t /root/Drone-SAM-Adapter/research_workspace/artifacts/auto_loop/logs/*_train.log 2>/dev/null | head -n 1 || true)
if [ -n "$latest_auto_train_log" ] && [ -f "$latest_auto_train_log" ]; then
  auto_mtime=$(stat -c %Y "$latest_auto_train_log" 2>/dev/null || echo 0)
  if [ "$auto_mtime" -gt "$train_mtime" ]; then
    train_mtime="$auto_mtime"
    train_tail=$(tail -n 1 "$latest_auto_train_log" 2>/dev/null || true)
  fi
fi
legacy_mtime=$(stat -c %Y /tmp/train_rrf_dsd_gen30_nccl_20260324.log 2>/dev/null || echo 0)
if [ "$legacy_mtime" -gt "$train_mtime" ]; then
  train_mtime="$legacy_mtime"
  train_tail=$(tail -n 1 /tmp/train_rrf_dsd_gen30_nccl_20260324.log 2>/dev/null || true)
fi
if [ -n "$active_work_dir" ] && [ -d "$active_work_dir" ]; then
  latest_ckpt=$(ls -1t "$active_work_dir"/epoch_*.pth "$active_work_dir"/best.pth 2>/dev/null | head -n 1 || true)
  if [ -n "$latest_ckpt" ]; then
    ckpt_mtime=$(stat -c %Y "$latest_ckpt" 2>/dev/null || echo 0)
    if [ "$ckpt_mtime" -gt "$train_mtime" ]; then
      train_mtime="$ckpt_mtime"
    fi
  fi
fi
if ps -eo cmd= | awk '/scripts\/auto_train_eval_loop.py/ {found=1} END{exit(found?0:1)}'; then
  sched_mtime=$(date +%s)
else
  sched_mtime=$(stat -c %Y /tmp/auto_loop_scheduler_20260324.log 2>/dev/null || echo 0)
fi
leader_mtime=$(stat -c %Y /root/Drone-SAM-Adapter/research_workspace/artifacts/auto_loop/leaderboard.md 2>/dev/null || echo 0)
history_lines=$(wc -l < /root/Drone-SAM-Adapter/research_workspace/artifacts/auto_loop/history.jsonl 2>/dev/null || echo 0)
pending_recipe_cnt=$(python3 - <<'PYEOF' 2>/dev/null || echo 0
import json
from pathlib import Path

repo = Path("/root/Drone-SAM-Adapter")
cfg_path = repo / "research_workspace/plans/auto_loop_config.json"
hist_path = repo / "research_workspace/artifacts/auto_loop/history.jsonl"

if not cfg_path.exists():
    print(0)
    raise SystemExit

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
recipes = [item.get("name") for item in cfg.get("recipes", []) if item.get("name")]
done = set()
if hist_path.exists():
    for line in hist_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("status") == "done" and obj.get("recipe"):
            done.add(obj["recipe"])

pending = [name for name in recipes if name not in done]
print(len(pending))
PYEOF
)
train_tail_b64=$(printf '%s' "$train_tail" | base64 -w0 || true)
sched_tail_b64=$(tail -n 1 /tmp/auto_loop_scheduler_20260324.log 2>/dev/null | base64 -w0 || true)
printf 'train_cnt=%s\n' "$train_cnt"
printf 'loop_cnt=%s\n' "$loop_cnt"
printf 'df_avail_gb=%s\n' "$df_avail_gb"
printf 'gpu_util=%s\n' "$gpu_util"
printf 'active_work_dir=%s\n' "$active_work_dir"
printf 'train_mtime=%s\n' "$train_mtime"
printf 'sched_mtime=%s\n' "$sched_mtime"
printf 'leader_mtime=%s\n' "$leader_mtime"
printf 'history_lines=%s\n' "$history_lines"
printf 'pending_recipe_cnt=%s\n' "$pending_recipe_cnt"
printf 'train_tail_b64=%s\n' "$train_tail_b64"
printf 'sched_tail_b64=%s\n' "$sched_tail_b64"
EOF
)
  ssh_run "bash -lc $(printf '%q' "$cmd")"
}

decode_b64() {
  local value="${1:-}"
  if [[ -z "$value" ]]; then
    echo ""
    return
  fi
  echo "$value" | base64 -d 2>/dev/null || echo ""
}

auto_recover_remote_loop() {
  local recover_cmd
  recover_cmd=$(cat <<'EOF'
nohup bash -lc '
cd /root/Drone-SAM-Adapter
/root/miniconda3/envs/torch211/bin/python scripts/auto_train_eval_loop.py \
  --config research_workspace/plans/auto_loop_config.json \
  --execute \
  --max-cycles 3
' > /tmp/auto_loop_autorecover.log 2>&1 &
echo $!
EOF
)
  ssh_run "bash -lc $(printf '%q' "$recover_cmd")"
}

auto_cleanup_remote_disk() {
  local cleanup_cmd
  cleanup_cmd=$(cat <<'EOF'
set -euo pipefail
cd /root/Drone-SAM-Adapter

target_free_gb="${TARGET_FREE_GB:-12}"
get_free_gb() {
  df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}'
}

# 识别当前正在训练的 work_dir，避免误删活跃目录中的 checkpoint
active_work_dir=$(ps -eo cmd= | awk '
/segmentation\/train_cacaf.py/ {
  for (i=1; i<=NF; i++) {
    if ($i=="--work-dir") {
      print $(i+1)
    }
  }
}' | tail -n 1)

# 历史遗留目录可直接清理
rm -rf work_dirs/rrf_dsd_v2_gen30_nccl_ohemdice 2>/dev/null || true

# 读取 auto-loop 配置中的 work_dir_root（支持绝对/相对路径）
configured_work_dir_root=$(python3 - <<'PYEOF' 2>/dev/null || true
import json
from pathlib import Path

cfg_path = Path("/root/Drone-SAM-Adapter/research_workspace/plans/auto_loop_config.json")
if not cfg_path.exists():
    print("/root/Drone-SAM-Adapter/work_dirs/auto_loop")
    raise SystemExit

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
wd = str(cfg.get("work_dir_root", "work_dirs/auto_loop"))
p = Path(wd)
if not p.is_absolute():
    p = Path("/root/Drone-SAM-Adapter") / p
print(str(p))
PYEOF
)

# 保留 top-N 历史最优 run 的 best.pth，其余 run 删除全部 checkpoint（不删日志/结果文本）
keep_top_n="${KEEP_TOP_N_CHECKPOINTS:-2}"
mapfile -t keep_work_dirs < <(python3 - <<'PYEOF' 2>/dev/null || true
import json
from pathlib import Path

hist_path = Path("/root/Drone-SAM-Adapter/research_workspace/artifacts/auto_loop/history.jsonl")
keep_n = int(__import__("os").environ.get("KEEP_TOP_N_CHECKPOINTS", "2"))

rows = []
if hist_path.exists():
    for line in hist_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("status") != "done":
            continue
        wd = obj.get("work_dir")
        selected = obj.get("selected") or {}
        score = selected.get("score")
        if not wd or score is None:
            continue
        rows.append((float(score), wd))

rows.sort(key=lambda x: x[0], reverse=True)
for _, wd in rows[:keep_n]:
    print(wd)
PYEOF
)

is_keep_work_dir() {
  local target="$1"
  local k
  for k in "${keep_work_dirs[@]:-}"; do
    if [ "$target" = "$k" ]; then
      return 0
    fi
  done
  return 1
}

for root in "$configured_work_dir_root" "/root/Drone-SAM-Adapter/work_dirs/auto_loop"; do
  [ -d "$root" ] || continue
  for d in "$root"/gen_*; do
    [ -d "$d" ] || continue
    abs_d="$d"

    if [ -n "${active_work_dir:-}" ] && [ "$abs_d" = "$active_work_dir" ]; then
      continue
    fi

    if is_keep_work_dir "$abs_d"; then
      rm -f "$d"/epoch_*.pth
    else
      rm -f "$d"/epoch_*.pth "$d"/best.pth
    fi
  done
done

free_gb="$(get_free_gb)"

# 如果仍低于目标阈值，继续清理缓存
if [ "$free_gb" -lt "$target_free_gb" ]; then
  conda clean -a -y >/dev/null 2>&1 || true
  rm -rf /root/.cache/pip/* >/dev/null 2>&1 || true
  free_gb="$(get_free_gb)"
fi

# 如果仍低于目标阈值，瘦身大日志（保留末尾）
if [ "$free_gb" -lt "$target_free_gb" ]; then
  for f in \
    /tmp/auto_loop_autorecover.log \
    /tmp/auto_loop_scheduler_20260324.log \
    /tmp/train_rrf_dsd_gen30_nccl_20260324.log
  do
    [ -f "$f" ] || continue
    tmp="${f}.tailtmp"
    tail -n 2000 "$f" > "$tmp" 2>/dev/null || true
    mv "$tmp" "$f" 2>/dev/null || true
  done
  free_gb="$(get_free_gb)"
fi

printf '%s\n' "$free_gb"
EOF
)
  ssh_run "KEEP_TOP_N_CHECKPOINTS=${KEEP_TOP_N_CHECKPOINTS} TARGET_FREE_GB=${TARGET_FREE_GB} bash -lc $(printf '%q' "$cleanup_cmd")"
}

check_once() {
  local raw
  if ! raw="$(collect_remote_status 2>&1)"; then
    alert "[ERROR] heartbeat ssh failed: ${raw}"
    return 1
  fi

  local train_cnt=0 loop_cnt=0 df_avail_gb=0 gpu_util="NA"
  local active_work_dir=""
  local train_mtime=0 sched_mtime=0 leader_mtime=0 history_lines=0 pending_recipe_cnt=0
  local train_tail_b64="" sched_tail_b64=""

  while IFS='=' read -r key value; do
    case "$key" in
      train_cnt) train_cnt="$value" ;;
      loop_cnt) loop_cnt="$value" ;;
      df_avail_gb) df_avail_gb="$value" ;;
      gpu_util) gpu_util="$value" ;;
      active_work_dir) active_work_dir="$value" ;;
      train_mtime) train_mtime="$value" ;;
      sched_mtime) sched_mtime="$value" ;;
      leader_mtime) leader_mtime="$value" ;;
      history_lines) history_lines="$value" ;;
      pending_recipe_cnt) pending_recipe_cnt="$value" ;;
      train_tail_b64) train_tail_b64="$value" ;;
      sched_tail_b64) sched_tail_b64="$value" ;;
    esac
  done <<< "$raw"

  local now epoch_train_age epoch_sched_age
  now=$(date +%s)
  epoch_train_age=$(( now - train_mtime ))
  epoch_sched_age=$(( now - sched_mtime ))

  local train_tail sched_tail
  train_tail="$(decode_b64 "$train_tail_b64")"
  sched_tail="$(decode_b64 "$sched_tail_b64")"

  {
    echo "timestamp=$(ts)"
    echo "train_cnt=${train_cnt}"
    echo "loop_cnt=${loop_cnt}"
    echo "df_avail_gb=${df_avail_gb}"
    echo "gpu_util=${gpu_util}"
    echo "active_work_dir=${active_work_dir}"
    echo "train_log_age_sec=${epoch_train_age}"
    echo "sched_log_age_sec=${epoch_sched_age}"
    echo "history_lines=${history_lines}"
    echo "pending_recipe_cnt=${pending_recipe_cnt}"
    echo "train_tail=${train_tail}"
    echo "sched_tail=${sched_tail}"
  } > "${STATUS_FILE}"

  log "heartbeat ok | train=${train_cnt} loop=${loop_cnt} free=${df_avail_gb}G gpu=${gpu_util} history=${history_lines}"

  if [[ "${df_avail_gb}" =~ ^[0-9]+$ ]] && (( df_avail_gb < MIN_FREE_GB )); then
    alert "[WARN] remote disk low: avail=${df_avail_gb}G (<${MIN_FREE_GB}G)"
    if (( AUTO_DISK_CLEANUP == 1 )); then
      local after_gb
      if after_gb="$(auto_cleanup_remote_disk 2>/dev/null)"; then
        alert "[AUTO-CLEANUP] remote disk cleanup done: ${df_avail_gb}G -> ${after_gb}G"
      else
        alert "[ERROR] auto disk cleanup failed"
      fi
    fi
  fi

  if (( train_cnt > 0 && epoch_train_age > STALE_SEC )); then
    alert "[WARN] train log stale: ${epoch_train_age}s (> ${STALE_SEC}s)"
  fi

  if (( loop_cnt > 0 && epoch_sched_age > STALE_SEC )); then
    alert "[WARN] scheduler log stale: ${epoch_sched_age}s (> ${STALE_SEC}s)"
  fi

  if (( train_cnt == 0 && loop_cnt == 0 )); then
    alert "[WARN] no train/scheduler process found on remote"
    if (( AUTO_RECOVER == 1 )) && [[ "${pending_recipe_cnt}" =~ ^[0-9]+$ ]] && (( pending_recipe_cnt > 0 )); then
      local new_pid
      if new_pid="$(auto_recover_remote_loop 2>/dev/null)"; then
        alert "[AUTO-RECOVER] started remote auto loop pid=${new_pid}"
      else
        alert "[ERROR] auto recover failed to start remote auto loop"
      fi
    else
      log "no pending recipe, skip auto recover"
    fi
  fi
}

run_loop() {
  log "heartbeat daemon started | interval=${CHECK_INTERVAL_SEC}s min_free=${MIN_FREE_GB}G target_free=${TARGET_FREE_GB}G keep_top=${KEEP_TOP_N_CHECKPOINTS} stale=${STALE_SEC}s auto_recover=${AUTO_RECOVER}"
  while true; do
    check_once || true
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_daemon() {
  if is_running; then
    echo "heartbeat already running (pid=$(cat "${PID_FILE}"))"
    return 0
  fi

  nohup "$0" run >> "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  sleep 1
  echo "heartbeat started (pid=$(cat "${PID_FILE}"))"
}

stop_daemon() {
  if ! is_running; then
    echo "heartbeat not running"
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "heartbeat stopped"
}

status_daemon() {
  if is_running; then
    echo "heartbeat running (pid=$(cat "${PID_FILE}"))"
  else
    echo "heartbeat not running"
  fi

  if [[ -f "${STATUS_FILE}" ]]; then
    echo "----- latest status -----"
    cat "${STATUS_FILE}"
  fi
}

case "${1:-}" in
  run)
    run_loop
    ;;
  start)
    start_daemon
    ;;
  stop)
    stop_daemon
    ;;
  restart)
    stop_daemon
    start_daemon
    ;;
  status)
    status_daemon
    ;;
  once)
    check_once
    ;;
  logs)
    tail -n 120 "${LOG_FILE}" 2>/dev/null || true
    ;;
  alerts)
    tail -n 120 "${ALERT_FILE}" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|once|logs|alerts}"
    exit 1
    ;;
esac
