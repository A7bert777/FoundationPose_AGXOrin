#!/usr/bin/env bash
# FoundationPose_AGXOrin 环境检查：CUDA / 编译器 / 扩展 / 包 / 权重 / 数据。
#
# 与 py/selfcheck.py 互补：
#   * selfcheck.py 在 **venv 内**做 import 与资源检查（Python 视角）；
#   * 本脚本在 **shell 层**检查系统工具链（nvcc / cmake / gcc）、原生扩展 .so、
#     TensorRT、关键包版本、以及运行所需的环境变量，便于初始化排障。
#
# 用法：./sh/check_env.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
PY="$VENV/bin/python"
SRC="$ROOT/framework_src"

# 让 `import estimater` / `import datareader` 能找到官方源码（与 fp_infer 一致）
export PYTHONPATH="$SRC:${PYTHONPATH:-}"

pass=0
fail=0
warn=0

ok()   { echo "  ✅ $1"; pass=$((pass+1)); }
bad()  { echo "  ❌ $1"; fail=$((fail+1)); }
meh()  { echo "  ⚠️  $1"; warn=$((warn+1)); }

echo "=============================================="
echo " FoundationPose_AGXOrin 环境检查"
echo " 根目录: $ROOT"
echo "=============================================="

# ---------- 1. 系统工具链 ----------
echo "[1] 系统工具链"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  ok "nvcc: $("$CUDA_HOME/bin/nvcc" --version | grep -oE 'release [0-9.]+' | head -1)"
else
  bad "未找到 nvcc（$CUDA_HOME/bin/nvcc），pytorch3d/nvdiffrast 无法编译"
fi
for t in cmake make g++; do
  if command -v "$t" >/dev/null 2>&1; then
    ok "$t: $(command -v "$t")"
  else
    bad "缺少 $t（pip: cmake/ninja；apt: build-essential）"
  fi
done
if [[ "$(uname -m)" == "aarch64" ]]; then
  ok "架构 aarch64（Jetson）"
else
  meh "当前架构 $(uname -m)，非 Jetson aarch64"
fi

# ---------- 2. venv ----------
echo "[2] Python 虚拟环境"
if [[ -x "$PY" ]]; then
  ok "venv: $VENV（$("$PY" -V 2>&1)）"
else
  bad "未找到 venv（$VENV），请先执行 ./sh/setup.sh"
  echo "=============================================="
  echo " 结论：venv 缺失，无法继续。"
  exit 1
fi

# ---------- 3. 原生扩展 .so ----------
echo "[3] 原生扩展"
check_import() {  # 模块、说明
  if "$PY" -c "import $1" >/dev/null 2>&1; then
    ok "$2（$1）"
  else
    bad "$2（$1）不可导入 —— 见 logs/build_*.log；重试：./sh/build_ext.sh $1"
  fi
}
check_import "pytorch3d" "pytorch3d"
check_import "nvdiffrast.torch" "nvdiffrast"
if "$PY" -c "import estimater" >/dev/null 2>&1; then
  ok "官方源码 estimater（$SRC）"
else
  bad "官方源码 estimater 不可导入（$SRC），请确认 framework_src 完整"
fi

MYCPP_SO="$(ls "$SRC"/mycpp/build/mycpp*.so 2>/dev/null | head -1 || true)"
if [[ -n "$MYCPP_SO" ]]; then
  if "$PY" -c "import sys; sys.path.insert(0,'$SRC/mycpp/build'); import mycpp" >/dev/null 2>&1; then
    ok "mycpp（$(basename "$MYCPP_SO")）"
  else
    bad "mycpp .so 存在但导入失败：$MYCPP_SO"
  fi
else
  bad "mycpp 未构建（$SRC/mycpp/build）—— 重试：./sh/build_ext.sh mycpp"
fi
# ---------- 4. 关键包版本 ----------
echo "[4] 关键包"
"$PY" - <<'PY'
import importlib
mods = ["torch", "torchvision", "numpy", "scipy", "open3d", "trimesh",
        "imageio", "cv2", "kornia", "ruamel.yaml", "warp", "tensorrt"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", getattr(mod, "version", "?"))
        print(f"  ✅ {m:14s} {v}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  {m:14s} 不可用：{type(e).__name__}")
PY

# ---------- 5. CUDA / 设备 ----------
echo "[5] CUDA / 设备"
"$PY" - <<'PY'
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  ✅ CUDA 可用 | {torch.cuda.get_device_name(0)} | "
          f"显存 {total/2**30:.1f}GiB（空闲 {free/2**30:.1f}GiB） | "
          f"CUDA {torch.version.cuda}")
else:
    print("  ❌ CUDA 不可用（torch 未编译 CUDA 或驱动/运行时缺失）")
PY

# ---------- 6. 权重与示例数据 ----------
echo "[6] 权重与示例数据"
SCORER="$ROOT/weights/2024-01-11-20-02-45/model_best.pth"
REFINER="$ROOT/weights/2023-10-28-18-33-37/model_best.pth"
DEMO="$ROOT/demo_data/mustard0/mesh/textured_simple.obj"
[[ -f "$SCORER"  ]] && ok "Scorer 权重"  || bad "缺少 Scorer 权重，请执行 ./sh/download.sh"
[[ -f "$REFINER" ]] && ok "Refiner 权重" || bad "缺少 Refiner 权重，请执行 ./sh/download.sh"
[[ -f "$DEMO"    ]] && ok "示例数据 demo_data/mustard0" || meh "示例数据缺失（可选），请执行 ./sh/download.sh"

# ---------- 7. 运行库路径 ----------
echo "[7] 运行库路径"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  meh "LD_LIBRARY_PATH 已设置（运行 sh/run.sh 会自动设置，无需手动）"
else
  ok "LD_LIBRARY_PATH 未设置（sh/run.sh 会自动注入 CUDA 运行库路径）"
fi

echo "=============================================="
echo " 结论：通过 $pass 项，失败 $fail 项，提示 $warn 项"
if [[ $fail -eq 0 ]]; then
  echo " 环境就绪 ✅"
else
  echo " 存在失败项，请按上方提示修复 ❌"
fi
echo "=============================================="
exit $([[ $fail -eq 0 ]] && echo 0 || echo 1)