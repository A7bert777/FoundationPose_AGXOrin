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

mkdir -p "$LOG" "$ROOT/models" "$ROOT/assets" "$ROOT/inputimage" \
         "$ROOT/outputimage" "$ROOT/outputs" "$ROOT/weights" "$ROOT/demo_data"

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
  # 说明：部分 JetPack 系统的 python3.10 未自带 ensurepip（缺 python3.10-venv 包）。
  # 此时 `python3.10 -m venv` 会失败。这里先用标准方式创建；若失败则回退到
  # `--without-pip` + get-pip.py 引导 pip，全程不需要 sudo。
  if python3.10 -m venv "$VENV" 2>/dev/null && [ -x "$PY" ]; then
    echo "      已创建：$VENV（标准方式）"
  else
    echo "      ensurepip 不可用，回退到 --without-pip + get-pip.py ..."
    rm -rf "$VENV"
    python3.10 -m venv --without-pip "$VENV" || {
      echo "[错误] 创建 venv 失败，请安装 python3.10-venv 后重试。"; exit 1; }
    GETPIP="$LOG/get-pip.py"
    if [ ! -s "$GETPIP" ]; then
      curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP" || {
        echo "[错误] 下载 get-pip.py 失败，请检查网络。"; exit 1; }
    fi
    "$PY" "$GETPIP" > "$LOG/getpip.log" 2>&1 || { tail -10 "$LOG/getpip.log"; exit 1; }
    echo "      已创建：$VENV（--without-pip + get-pip.py 引导）"
  fi
  echo "      注意：这是本项目专属 venv，未复用任何其它项目环境"
fi

export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

# ---------- 2. torch ----------
if "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "[2/7] torch 已就绪: $("$PY" -c 'import torch; print(torch.__version__)')"
else
  echo "[2/7] 安装 NVIDIA 定制 PyTorch（Jetson aarch64, JetPack 6.x）..."
  "$PY" -m pip install --upgrade pip wheel > "$LOG/pip.log" 2>&1 || true
  "$PY" -m pip install --no-cache-dir \
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
    "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl" \
    >> "$LOG/pip.log" 2>&1 \
    || { echo "[提示] 自动安装失败，请参考 NVIDIA Jetson PyTorch 论坛贴手动安装 torch/torchvision。"; exit 1; }
fi

# ---------- 3. Python 依赖 ----------
echo "[3/7] 安装 FoundationPose Python 依赖（日志：logs/deps.log）..."
"$PY" -m pip install --no-cache-dir -r "$ROOT/requirements.txt" \
  > "$LOG/deps.log" 2>&1 || { tail -30 "$LOG/deps.log"; exit 1; }
# 额外：gdown（权重下载）、hf 镜像等
"$PY" -m pip install --no-cache-dir gdown >> "$LOG/deps.log" 2>&1 || true
echo "      依赖安装完成"

# ---------- 4. 官方源码 ----------
# framework_src/ 是 NVlabs/FoundationPose 的 git clone，本仓库不随附（授权 + 体积），
# 在此自动获取。必须在第 5 步编译原生扩展之前就位（build_ext.sh 依赖它的 mycpp/）。
if [ ! -f "$ROOT/framework_src/estimater.py" ]; then
  echo "[4/7] 获取官方 FoundationPose 源码（framework_src）..."
  git clone https://github.com/NVlabs/FoundationPose.git "$ROOT/framework_src" \
    > "$LOG/clone_framework_src.log" 2>&1 || {
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