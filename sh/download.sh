#!/usr/bin/env bash
# 下载 FoundationPose 所需的本地资源（支持断点续传，重复执行即可）：
#   1) weights/2024-01-11-20-02-45/  Scorer 权重（model_best.pth + config.yml）
#   2) weights/2023-10-28-18-33-37/  Refiner 权重（model_best.pth + config.yml）
#   3) demo_data/mustard0/           官方示例场景（RGB-D 视频 + 网格）
#
# 官方权重托管在 Google Drive 文件夹，直接 curl 无法下载；本脚本按以下顺序尝试：
#   a) 本地已有 -> 跳过
#   b) HuggingFace 镜像（若配置了 FP_WEIGHTS_BASE 环境变量）/ hf-mirror
#   c) gdown 拉取 Google Drive 文件夹（需要 pip install gdown）
#
# 可通过环境变量覆盖：
#   FP_WEIGHTS_BASE   指向一个直链目录（形如 https://host/fp_weights）
#                     期望其下为 <BASE>/2024-01-11-20-02-45/model_best.pth 等
#   HF_ENDPOINT       HuggingFace 镜像（默认 https://hf-mirror.com）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv310/bin/python"
mkdir -p "$ROOT/weights" "$ROOT/demo_data" "$ROOT/logs"

SCORER_STAMP="2024-01-11-20-02-45"
REFINER_STAMP="2023-10-28-18-33-37"

# Google Drive 文件夹 ID（来自官方 readme「Data prepare」）
GDRIVE_WEIGHTS_ID="1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
GDRIVE_DEMO_ID="1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP"

echo "=============================================="
echo " FoundationPose_AGXOrin 资源下载"
echo " 项目根目录: $ROOT"
echo "=============================================="

# ------------------------------------------------------------------ #
# 工具函数：下载单个文件（多源回退 + 断点续传 + 最小字节校验）
# ------------------------------------------------------------------ #
fetch() {  # fetch <目标文件> <最小字节> <候选URL...>
  local out="$1" min="$2"; shift 2
  if [[ -f "$out" ]] && (( $(stat -c%s "$out") >= min )); then
    echo "[跳过] 已存在：$out ($(du -h "$out" | cut -f1))"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  for url in "$@"; do
    echo "[下载] $url"
    if curl -fL --retry 5 --retry-delay 3 -C - -o "$out" "$url"; then
      if (( $(stat -c%s "$out") >= min )); then
        echo "[完成] $out ($(du -h "$out" | cut -f1))"
        return 0
      fi
      echo "[警告] 文件偏小，尝试下一个源"
    fi
  done
  echo "[失败] 无法下载：$out"
  return 1
}

# ------------------------------------------------------------------ #
# 权重：优先使用直链镜像（FP_WEIGHTS_BASE），否则回退 gdown
# ------------------------------------------------------------------ #
download_weights() {
  local stamp="$1" label="$2"
  local dir="$ROOT/weights/$stamp"
  local ckpt="$dir/model_best.pth"
  local cfg="$dir/config.yml"

  if [[ -f "$ckpt" && -f "$cfg" ]]; then
    echo "[跳过] $label 权重已存在：$dir"
    return 0
  fi

  if [[ -n "${FP_WEIGHTS_BASE:-}" ]]; then
    fetch "$ckpt" 1000000 "$FP_WEIGHTS_BASE/$stamp/model_best.pth" || true
    fetch "$cfg"  100         "$FP_WEIGHTS_BASE/$stamp/config.yml"    || true
    if [[ -f "$ckpt" && -f "$cfg" ]]; then
      echo "[完成] $label 权重（直链镜像）"
      return 0
    fi
  fi

  echo "[提示] $label 权重未就绪，尝试用 gdown 拉取 Google Drive 文件夹 ..."
  if "$PY" -c "import gdown" >/dev/null 2>&1; then
    "$PY" -m gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_WEIGHTS_ID" \
        -O "$ROOT/weights" 2>&1 | tail -5 || true
  else
    echo "[警告] 未安装 gdown，无法自动下载权重。"
    echo "       请任选其一："
    echo "         pip install gdown && ./sh/download.sh"
    echo "         或手动从官方 Google Drive 下载后放到 $dir/"
    echo "         或设置 FP_WEIGHTS_BASE 指向直链镜像目录后重跑本脚本"
  fi
}

download_weights "$SCORER_STAMP"  "Scorer"
download_weights "$REFINER_STAMP" "Refiner"

# ------------------------------------------------------------------ #
# 示例数据（demo_data/mustard0）
# ------------------------------------------------------------------ #
DEMO_DIR="$ROOT/demo_data/mustard0"
DEMO_MESH="$DEMO_DIR/mesh/textured_simple.obj"
if [[ -f "$DEMO_MESH" ]]; then
  echo "[跳过] 示例数据已存在：$DEMO_DIR"
else
  echo "[提示] 示例数据未就绪，尝试用 gdown 拉取 ..."
  if "$PY" -c "import gdown" >/dev/null 2>&1; then
    "$PY" -m gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_DEMO_ID" \
        -O "$ROOT/demo_data" 2>&1 | tail -5 || true
  else
    echo "[警告] 未安装 gdown，跳过示例数据下载。"
  fi
fi

# ------------------------------------------------------------------ #
# 汇总
# ------------------------------------------------------------------ #
echo
echo "资源状态："
for stamp in "$SCORER_STAMP" "$REFINER_STAMP"; do
  d="$ROOT/weights/$stamp"
  if [[ -f "$d/model_best.pth" && -f "$d/config.yml" ]]; then
    echo "  [OK]   weights/$stamp  ($(du -sh "$d" | cut -f1))"
  else
    echo "  [缺失] weights/$stamp  —— 见上方提示"
  fi
done
if [[ -f "$DEMO_MESH" ]]; then
  echo "  [OK]   demo_data/mustard0"
else
  echo "  [缺失] demo_data/mustard0 —— 见上方提示"
fi