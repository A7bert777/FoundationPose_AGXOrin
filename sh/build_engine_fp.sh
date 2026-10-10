#!/usr/bin/env bash
# 用 trtexec 把 FoundationPose 导出的 ONNX 构建为 TensorRT engine（FP32）
#
# 用法:
#   bash build_engine_fp.sh <onnx> [precision] [workspace_MB]
# 例:
#   bash build_engine_fp.sh models/scorer.onnx  fp32 4096
#   bash build_engine_fp.sh models/refiner.onnx fp32 4096
#
# 说明:
#   - 精度固定 FP32：不加 --fp16/--int8，显式 --precisionConstraints=obey
#     （位姿回归对数值敏感，FP16 会明显掉精度）
#   - 输出的 engine 与 ONNX 同目录，同名 .engine
#   - ONNX 由 py/export_onnx_fp.py 生成
set -euo pipefail
cd "$(dirname "$0")/.."

ONNX="${1:?用法: build_engine_fp.sh <onnx> [fp32] [workspace_MB]}"
PREC="${2:-fp32}"
WS_MB="${3:-4096}"

[ -f "$ONNX" ] || { echo "[错误] 找不到 $ONNX"; exit 1; }
[ "$PREC" = "fp32" ] || { echo "[错误] 本脚本只构建 FP32（当前传入 $PREC）"; exit 1; }

TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -x "$TRTEXEC" ] || TRTEXEC="$(command -v trtexec)"
[ -n "${TRTEXEC:-}" ] || { echo "[错误] 找不到 trtexec"; exit 1; }

ENGINE="${ONNX%.onnx}.engine"
LOG="logs/engine_$(basename "${ONNX%.onnx}")_${PREC}.log"

echo "======================================================================"
echo " 构建 TensorRT engine (FP32)"
echo "   ONNX      : $ONNX"
echo "   ENGINE    : $ENGINE"
echo "   workspace : ${WS_MB} MB"
echo "   trtexec   : $TRTEXEC"
echo "   日志      : $LOG"
echo "======================================================================"

START=$(date +%s)

"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --memPoolSize=workspace:${WS_MB} \
  --precisionConstraints=obey \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --verbose \
  > "$LOG" 2>&1 || {
    echo "[失败] trtexec 返回非 0，末尾日志："
    tail -40 "$LOG"
    exit 1
  }

END=$(date +%s)
DUR=$((END - START))

echo
echo "----------------------------------------------------------------------"
echo " 构建完成，耗时 ${DUR}s"
ls -lh "$ENGINE" 2>/dev/null || true
echo "----------------------------------------------------------------------"
grep -iE "Engine built in|Serialized engine|Total Host Walltime" "$LOG" | tail -6 || true
echo "耗时: ${DUR}s"
echo "$DUR" > "${ENGINE}.build_seconds"
echo "已写入 ${ENGINE}.build_seconds"