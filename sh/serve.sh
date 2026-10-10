#!/usr/bin/env bash
# FoundationPose 常驻服务管理：模型/权重只加载一次，之后每次请求复用。
#
# 用法：
#   ./sh/serve.sh start [额外参数]   # 后台启动服务（默认不超时退出）
#   ./sh/serve.sh stop               # 释放模型与显存
#   ./sh/serve.sh restart            # 重启
#   ./sh/serve.sh status             # 查看服务状态与显存
#   ./sh/serve.sh log                # 跟踪服务日志
#   ./sh/serve.sh fore [额外参数]    # 前台启动（Ctrl+C 退出）
#
# 启动后正常使用 ./sh/run.sh，参数完全不变：
#   ./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj
#
# 可选参数（透传给 fp_server.py）：
#   --idle-timeout 1800     空闲 30 分钟自动退出并释放显存
#   --warmup                启动时预热一次各网络
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"

RUN_DIR="$ROOT/run"
SOCK="$RUN_DIR/fp.sock"
PIDFILE="$RUN_DIR/fp.pid"
LOGFILE="$RUN_DIR/fp.log"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./sh/setup.sh 初始化环境"
  exit 1
fi

export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "$RUN_DIR"

# 读取 pidfile 里的 PID（若进程仍存活则输出，否则返回空）
alive_pid() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

cmd_start() {
  if pid="$(alive_pid)"; then
    echo "[服务] 已在运行（PID=$pid，socket=$SOCK）"
    exit 0
  fi
  rm -f "$SOCK" "$PIDFILE"
  echo "[服务] 正在后台启动（首次需加载模型，可能数十秒）..."
  nohup "$VENV/bin/python" "$ROOT/py/fp_server.py" \
      --socket "$SOCK" "$@" >"$LOGFILE" 2>&1 &
  local pid=$!
  echo "$pid" >"$PIDFILE"

  # 等待 socket 就绪（最多 180s）
  for _ in $(seq 1 360); do
    if [[ -S "$SOCK" ]]; then
      echo "[服务] 就绪（PID=$pid）"
      echo "       日志：./sh/serve.sh log"
      echo "       现在可直接用：./sh/run.sh --video ... --mesh ... （示例）"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[错误] 服务启动失败，日志末尾："
      tail -n 30 "$LOGFILE" || true
      exit 1
    fi
    sleep 0.5
  done
  echo "[警告] 等待 socket 超时，请查看日志：./sh/serve.sh log"
}

cmd_stop() {
  local pid
  if ! pid="$(alive_pid)"; then
    if [[ -S "$SOCK" ]]; then
      echo "[服务] 未发现服务进程，清理残留 socket"
      rm -f "$SOCK"
    else
      echo "[服务] 未在运行"
    fi
    rm -f "$PIDFILE"
    return 0
  fi

  echo "[服务] 正在停止（PID=$pid）..."
  # 优先优雅退出：发 stop 请求，让服务自己释放显存
  if [[ -S "$SOCK" ]]; then
    "$VENV/bin/python" - "$SOCK" <<'PY' 2>/dev/null || true
import json, socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(sys.argv[1])
    with s, s.makefile("rwb") as f:
        f.write((json.dumps({"cmd": "stop"}) + "\n").encode())
        f.flush()
        f.readline()
except OSError:
    pass
PY
  fi

  for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[服务] 优雅退出超时，强制结束"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$pid" 2>/dev/null || true
  fi

  rm -f "$SOCK" "$PIDFILE"
  echo "[服务] 已停止，模型与显存已释放"
}

cmd_status() {
  if pid="$(alive_pid)"; then
    echo "[服务] 运行中（PID=$pid）"
    [[ -S "$SOCK" ]] && echo "       socket: $SOCK" || echo "       socket: 缺失（可能正在启动）"
    if [[ -S "$SOCK" ]]; then
      "$VENV/bin/python" - "$SOCK" <<'PY' 2>/dev/null || true
import json, socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(sys.argv[1])
    with s, s.makefile("rwb") as f:
        f.write((json.dumps({"cmd": "ping"}) + "\n").encode())
        f.flush()
        r = json.loads(f.readline().decode())
    print(f"       设备: {r.get('device')} | 已运行 {r.get('uptime_s')}s")
except Exception as e:
    print(f"       ping 失败: {e}")
PY
    fi
  else
    echo "[服务] 未运行（此时 ./sh/run.sh 会自动本地加载模型）"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
  fi
}

cmd_log() {
  [[ -f "$LOGFILE" ]] || { echo "[服务] 暂无日志（$LOGFILE）"; exit 0; }
  tail -n 40 -f "$LOGFILE"
}

cmd_fore() {
  if pid="$(alive_pid)"; then
    echo "[错误] 已有服务在运行（PID=$pid），请先 ./sh/serve.sh stop"
    exit 1
  fi
  rm -f "$SOCK" "$PIDFILE"
  echo "[服务] 前台启动，Ctrl+C 退出并释放显存"
  exec "$VENV/bin/python" "$ROOT/py/fp_server.py" --socket "$SOCK" "$@"
}

case "${1:-}" in
  start)   shift; cmd_start "$@" ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; shift || true; cmd_start "$@" ;;
  status)  cmd_status ;;
  log)     cmd_log ;;
  fore)    shift; cmd_fore "$@" ;;
  ""|-h|--help|help)
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "[错误] 未知命令：$1"
    echo "       可用：start | stop | restart | status | log | fore"
    exit 1
    ;;
esac