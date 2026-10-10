#!/usr/bin/env bash
# 等 pytorch3d 浅克隆完成 -> 从本地源码安装（避免再次走 GitHub 网络克隆）
set -u
cd /home/jetson/zhangtianqi/FoundationPose_AGXOrin
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="$PWD/venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY=./venv310/bin/python
LOG=logs/build_p3d.log
: > "$LOG"

echo "### 等待 pytorch3d 克隆结束 ..." | tee -a "$LOG"
for i in $(seq 1 240); do
  if ! pgrep -f "git clone.*pytorch3d" >/dev/null; then break; fi
  sleep 15
done
echo "### 克隆结束，开始安装 ..." | tee -a "$LOG"
du -sh /tmp/pytorch3d >> "$LOG" 2>&1

echo "### 安装 pytorch3d（本地源码, no-build-isolation）..." | tee -a "$LOG"
$PY -m pip install --no-cache-dir --no-build-isolation /tmp/pytorch3d >> "$LOG" 2>&1
echo "### 安装退出码: $?" | tee -a "$LOG"

echo "### 验证 import:" | tee -a "$LOG"
$PY -c "import pytorch3d;print('pytorch3d OK', pytorch3d.__version__)" >> "$LOG" 2>&1

echo "### P3D_DONE" | tee -a "$LOG"