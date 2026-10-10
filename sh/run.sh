#!/usr/bin/env bash
# FoundationPose_AGXOrin 运行入口：设置环境后调用客户端 fp_client.py
#
# 用法（与 SAM3_AGXOrin 风格一致）：
#   ./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj \
#               --out outputimage/mustard_track.png
#   ./sh/run.sh --video <scene_dir> --mesh <obj> --est-refine-iter 5 --track-refine-iter 2 --bench
#
# 行为：
#   若已执行 ./sh/serve.sh start（常驻服务在线）-> 请求转发给服务（模型只加载一次）
#   否则                                        -> 本地加载模型单次运行
#   两种方式输出完全一致。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./sh/setup.sh 初始化环境"
  exit 1
fi

# cuSPARSELt 等 NVIDIA 运行库（Jetson 上 torch 依赖，使用系统 CUDA 库）
export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}"
# Jetson 上避免碎片化；如显存紧张可改为 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

exec "$VENV/bin/python" "$ROOT/py/fp_client.py" "$@"