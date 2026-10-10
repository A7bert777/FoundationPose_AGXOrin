#!/usr/bin/env bash
# FoundationPose_AGXOrin 用 TensorRT engine 推理的运行入口（可选加速路径）
#
# 说明：
#   FoundationPose 的位姿估计计算图包含「可微光栅化渲染 + 迭代精修」，
#   与 SAM3 这种纯前向网络不同，TRT 化收益有限且工程复杂。因此本路径为**可选**：
#   仅把 Scorer / Refiner 的**网络前向**导出 ONNX 并用 TRT 执行渲染之外的张量运算，
#   渲染仍走 nvdiffrast。
#
#   若尚未构建 engine，本脚本会提示并可直接回退到 PyTorch 路径（sh/run.sh）。
#
# 用法（与 run.sh 参数完全一致）：
#   ./sh/run_trt.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj --out outputimage/out.png
#
# 额外开关（A/B 对比）：
#   --scorer-backend  pytorch    关闭 scorer engine，回退 PyTorch
#   --refiner-backend pytorch    关闭 refiner engine，回退 PyTorch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./sh/setup.sh 初始化环境"
  exit 1
fi

SB="$ROOT/models/scorer.engine"
RB="$ROOT/models/refiner.engine"
if [[ ! -f "$SB" && ! -f "$RB" ]]; then
  echo "[错误] 未找到任何 engine 文件（$SB / $RB）"
  echo "       请先执行： ./sh/build_engine_fp.sh"
  echo "       或直接使用 PyTorch 路径： ./sh/run.sh --video ... --mesh ..."
  exit 1
fi

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

exec "$VENV/bin/python" "$ROOT/py/fp_trt_infer.py" "$@"