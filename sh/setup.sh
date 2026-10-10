#!/usr/bin/env bash
# FoundationPose_AGXOrin 一键环境初始化（Jetson AGX Orin / JetPack, aarch64, Python 3.10）
#
# 步骤：
#   1. 创建 **本项目独立** 的 venv310（若已存在则复用；绝不复用其它项目 venv）
#   2. 检查/安装 NVIDIA 定制 PyTorch（Jetson 专用 wheel）
#   3. 安装 FoundationPose Python 依赖
#   4. 获取官方 FoundationPose 源码（framework_src，本仓库不随附）
#   5. 编译原生扩展（pytorch3d / nvdiffrast / mycpp）—— 见 sh/build_ext.sh
#   6. 下载权重与示例数据（Google Drive / 镜像）
#   7. 自检 import estimater
#
# 用法：./sh/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.10/site-packages"
LOG="$ROOT/logs"

# 清掉外部 PYTHONPATH（例如 source 过 ROS 的 /opt/ros/*/lib/python3.X/site-packages）。
# 本 venv 是 python3.10，若 PYTHONPATH 里混入别的 Python 版本的 site-packages，
# pip 会把那些包误判为「已安装」，import 时也可能被意外遮蔽。
if [ -n "${PYTHONPATH:-}" ]; then
  echo "[提示] 已忽略外部 PYTHONPATH=$PYTHONPATH（避免污染本项目 venv）"
  unset PYTHONPATH
fi

mkdir -p "$LOG" "$ROOT/models" "$ROOT/assets" "$ROOT/inputimage" \
         "$ROOT/outputimage" "$ROOT/outputs" "$ROOT/weights" "$ROOT/demo_data" \
         "$ROOT/wheels"

# run_logged：长任务输出同时打到终端和日志（内部用 pty，保证 pip/gdown 的进度条实时可见）
# shellcheck source=sh/_common.sh
. "$ROOT/sh/_common.sh"

echo "=============================================="
echo " FoundationPose_AGXOrin 环境初始化"
echo " 项目根目录: $ROOT"
echo "=============================================="

# ---------- 1. venv（本项目独立，使用 python3.10）----------
if [ -x "$PY" ]; then
  echo "[1/7] 复用已有 venv: $VENV"
else
  echo "[1/7] 创建独立 venv310（python3.10）..."
  if ! command -v python3.10 >/dev/null 2>&1; then
    echo "[错误] 未找到 python3.10，请先安装（如 sudo apt-get install -y python3.10 python3.10-venv）"
    exit 1
  fi
  # 说明：部分 JetPack 系统的 python3.10 未装 python3.10-venv 包，因而不带 ensurepip，
  # `python3.10 -m venv` 会失败并打印一大段 ensurepip 报错。这里先探测 ensurepip：
  # 有就用标准方式，没有就直接走 `--without-pip` + get-pip.py 引导 pip，全程不需要 sudo。
  if python3.10 -c "import ensurepip" >/dev/null 2>&1 \
     && python3.10 -m venv "$VENV" 2>/dev/null && [ -x "$PY" ]; then
    echo "      已创建：$VENV（标准方式）"
  else
    echo "      未装 python3.10-venv（缺 ensurepip），改用 --without-pip + get-pip.py 引导"
    rm -rf "$VENV"
    python3.10 -m venv --without-pip "$VENV" || {
      echo "[错误] 创建 venv 失败，请安装 python3.10-venv 后重试。"; exit 1; }
    GETPIP="$LOG/get-pip.py"
    if [ ! -s "$GETPIP" ]; then
      curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP" || {
        echo "[错误] 下载 get-pip.py 失败，请检查网络。"; exit 1; }
    fi
    run_logged "$LOG/getpip.log" "$PY" "$GETPIP" || { tail -10 "$LOG/getpip.log"; exit 1; }
    echo "      已创建：$VENV（--without-pip + get-pip.py 引导）"
  fi
  echo "      注意：这是本项目专属 venv，未复用任何其它项目环境"
fi

export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

# ---------- 2. torch ----------
# 注意：NVIDIA 已下线旧的直链
#   https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/...
# 现在访问该目录下任何文件都返回 HTTP 404（不是网络问题）。
# 因此改为两级来源：
#   a) 本地 wheels/ 目录里已下好的 wheel（离线可用，版本完全可控）——优先
#   b) Jetson 社区镜像 pypi.jetson-ai-lab.io（PEP 503 索引，含 jp6/cu126 构建）
TORCH_INDEX="${FP_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
PIP_OPTS=(--no-cache-dir --timeout 120 --retries 5)

# 装单个包：本地有 wheel 就用本地的，否则走镜像
install_one() {   # $1=import 名  $2=包名  $3=版本  $4=本地 wheel 通配
  if "$PY" -c "import $1" >/dev/null 2>&1; then
    echo "      $2 已安装，跳过"
    return 0
  fi
  local w
  for w in "$ROOT/wheels/"$4; do
    if [ -f "$w" ]; then
      echo "      从本地 wheel 安装 $2：$(basename "$w")"
      run_logged "$LOG/pip.log" "$PY" -m pip install "${PIP_OPTS[@]}" "$w" && return 0
      break
    fi
  done
  echo "      从镜像安装 $2==$3 ..."
  run_logged "$LOG/pip.log" "$PY" -m pip install "${PIP_OPTS[@]}" \
    --index-url "$TORCH_INDEX" "$2==$3" && return 0
  return 1
}

if "$PY" -c "import torch, torchvision" >/dev/null 2>&1; then
  echo "[2/7] torch 已就绪: $("$PY" -c 'import torch; print(torch.__version__)')"
else
  echo "[2/7] 安装 NVIDIA 定制 PyTorch（Jetson aarch64, JetPack 6.x）..."
  echo "      torch wheel 约 216MB，视网速可能需数分钟；下方会实时显示进度条"
  run_logged "$LOG/pip.log" "$PY" -m pip install --upgrade pip wheel || true

  ok=1
  install_one torch       torch       2.8.0  "torch-2.8.0-*.whl"       || ok=0
  install_one torchvision torchvision 0.23.0 "torchvision-0.23.0-*.whl" || ok=0

  if [ "$ok" = 0 ]; then
    echo "[错误] torch / torchvision 安装失败，详见 $LOG/pip.log"
    echo "       手动方案：下载对应 wheel 放进 $ROOT/wheels/ 后重跑本脚本即可（会自动优先使用本地 wheel）："
    echo "         torch-2.8.0-cp310-cp310-linux_aarch64.whl"
    echo "         torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
    echo "       镜像索引：$TORCH_INDEX"
    exit 1
  fi
fi

# ---------- 3. Python 依赖 ----------
echo "[3/7] 安装 FoundationPose Python 依赖（含 open3d / warp-lang，体积较大，日志：logs/deps.log）..."
run_logged "$LOG/deps.log" "$PY" -m pip install --no-cache-dir -r "$ROOT/requirements.txt" \
  || { tail -30 "$LOG/deps.log"; exit 1; }
# 额外：gdown（权重下载）、hf 镜像等
run_logged "$LOG/deps.log" "$PY" -m pip install --no-cache-dir gdown || true
echo "      依赖安装完成"

# ---------- 4. 官方源码 ----------
# framework_src/ 是 NVlabs/FoundationPose 的 git clone，本仓库不随附（授权 + 体积），
# 在此自动获取。必须在第 5 步编译原生扩展之前就位（build_ext.sh 依赖它的 mycpp/）。
if [ ! -f "$ROOT/framework_src/estimater.py" ]; then
  echo "[4/7] 获取官方 FoundationPose 源码（framework_src）..."
  run_logged "$LOG/clone_framework_src.log" \
    git clone https://github.com/NVlabs/FoundationPose.git "$ROOT/framework_src" || {
      echo "[错误] clone 失败，请检查网络。也可手动执行："
      echo "        git clone https://github.com/NVlabs/FoundationPose.git $ROOT/framework_src"
      exit 1; }
  # 切换到本项目验证过的官方版本；若不可用则退回默认分支（不视为致命错误）
  if git -C "$ROOT/framework_src" checkout -q a1b694b 2>/dev/null; then
    echo "      已切换到验证版本 a1b694b"
  else
    echo "      [提示] 未能切到 a1b694b，使用默认分支"
  fi
else
  echo "[4/7] framework_src 已存在，跳过 clone"
fi

# 官方代码从 framework_src/weights/ 读取权重，而权重放在顶层 weights/，故建软链接。
if [ ! -e "$ROOT/framework_src/weights" ] && [ ! -L "$ROOT/framework_src/weights" ]; then
  ln -s ../weights "$ROOT/framework_src/weights"
  echo "      已建立软链接 framework_src/weights -> ../weights"
fi

# ---------- 5. 原生扩展 ----------
echo "[5/7] 编译原生扩展（pytorch3d / nvdiffrast / mycpp）..."
bash "$ROOT/sh/build_ext.sh" || {
  echo "[警告] 原生扩展编译未完全成功，可稍后单独重试： bash sh/build_ext.sh <pytorch3d|nvdiffrast|mycpp>"
}

# ---------- 5. 资源 ----------
echo "[6/7] 下载权重与示例数据 ..."
bash "$ROOT/sh/download.sh" || true

# ---------- 6. 自检 ----------
echo "[7/7] 自检 import estimater ..."
"$PY" "$ROOT/py/selfcheck.py" || true

echo
echo "=============================================="
echo " 完成！示例："
echo "   ./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj"
echo "   ./sh/serve.sh start        # 常驻服务（模型只加载一次）"
echo "=============================================="