#!/usr/bin/env bash
# 构建 FoundationPose 所需的全部原生扩展（Jetson aarch64）：
#   1) pytorch3d    —— 从源码编译（需要 nvcc / CUDA_HOME）
#   2) nvdiffrast   —— 从源码编译（可微光栅化，渲染刚体网格）
#   3) mycpp        —— FoundationPose 自带的 C++ 位姿聚类扩展（Boost/Eigen/pybind11）
#
# 说明：
#   * 全程在 **本项目的 venv310** 内进行，绝不触碰其它项目的虚拟环境。
#   * 需要系统具备：nvcc、cmake、make/g++、以及 eigen / boost 开发库。
#     若缺少可先：sudo apt-get install -y libeigen3-dev libboost-all-dev cmake build-essential
#   * BundleSDF 的 mycuda（model-free NeRF 路径）默认跳过，见文末说明。
#
# 用法：
#   ./sh/build_ext.sh             # 构建全部
#   ./sh/build_ext.sh pytorch3d   # 只构建某一个（pytorch3d | nvdiffrast | mycpp）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
PY="$VENV/bin/python"
SRC="$ROOT/framework_src"
LOG="$ROOT/logs"
mkdir -p "$LOG"

if [[ ! -x "$PY" ]]; then
  echo "[错误] 未找到 venv：$VENV，请先执行 ./sh/setup.sh"
  exit 1
fi

# ---- 环境变量：让源码编译能看到 CUDA 与本 venv 的 torch ----
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.7"          # Jetson AGX Orin = SM 8.7
export MAX_JOBS="${MAX_JOBS:-4}"           # Orin 上限制并行，避免内存打爆
export CMAKE_PREFIX_PATH="$VENV:${CMAKE_PREFIX_PATH:-}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "[警告] 未找到 nvcc（$CUDA_HOME/bin/nvcc），pytorch3d/nvdiffrast 将无法编译。"
  echo "       请确认已安装 CUDA toolkit 并设置 CUDA_HOME。"
fi

TARGETS="${1:-all}"
want() { [[ "$TARGETS" == "all" || "$TARGETS" == "$1" ]]; }

# ------------------------------------------------------------------ #
# 1) pytorch3d
# ------------------------------------------------------------------ #
build_pytorch3d() {
  echo "======================================================================"
  echo " [1/3] 编译 pytorch3d ..."
  echo "======================================================================"
  "$PY" -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git" \
    > "$LOG/build_pytorch3d.log" 2>&1 || {
      echo "[失败] pytorch3d 编译失败，末尾日志："
      tail -30 "$LOG/build_pytorch3d.log"
      return 1
    }
  echo "[完成] pytorch3d：$("$PY" -c 'import pytorch3d; print(pytorch3d.__version__)' 2>/dev/null || echo '?')"
}

# ------------------------------------------------------------------ #
# 2) nvdiffrast
# ------------------------------------------------------------------ #
build_nvdiffrast() {
  echo "======================================================================"
  echo " [2/3] 编译 nvdiffrast ..."
  echo "======================================================================"
  "$PY" -m pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git" \
    > "$LOG/build_nvdiffrast.log" 2>&1 || {
      echo "[失败] nvdiffrast 编译失败，末尾日志："
      tail -30 "$LOG/build_nvdiffrast.log"
      return 1
    }
  echo "[完成] nvdiffrast：$("$PY" -c 'import nvdiffrast; print(getattr(nvdiffrast,"__version__","ok"))' 2>/dev/null || echo '?')"
}

# ------------------------------------------------------------------ #
# 3) mycpp（FoundationPose 自带 C++ 扩展）
# ------------------------------------------------------------------ #
build_mycpp() {
  echo "======================================================================"
  echo " [3/3] 编译 mycpp（位姿聚类扩展）..."
  echo "======================================================================"
  local B="$SRC/mycpp/build"
  rm -rf "$B"
  mkdir -p "$B"
  (
    cd "$B"
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE="$PY" \
      -DPython3_ROOT_DIR="$VENV" \
      -DPYBIND11_PYTHON_EXECUTABLE="$PY" \
      > "$LOG/build_mycpp_cmake.log" 2>&1 || { tail -30 "$LOG/build_mycpp_cmake.log"; exit 1; }
    make -j"${MAX_JOBS}" \
      > "$LOG/build_mycpp_make.log" 2>&1 || { tail -30 "$LOG/build_mycpp_make.log"; exit 1; }
  ) || { echo "[失败] mycpp 编译失败，见 logs/build_mycpp_*.log"; return 1; }

  local so
  so="$(ls "$B"/mycpp*.so 2>/dev/null | head -1 || true)"
  if [[ -n "$so" ]]; then
    echo "[完成] mycpp：$(basename "$so")"
  else
    echo "[警告] 未找到 mycpp*.so"
    return 1
  fi
}

rc=0
if want pytorch3d;  then build_pytorch3d  || rc=1; fi
if want nvdiffrast; then build_nvdiffrast || rc=1; fi
if want mycpp;      then build_mycpp      || rc=1; fi

echo
if [[ $rc -eq 0 ]]; then
  echo "======================================================================"
  echo " 所有原生扩展构建完成 ✅"
  echo " 下一步：./venv310/bin/python py/selfcheck.py"
  echo "======================================================================"
else
  echo "======================================================================"
  echo " 部分扩展构建失败 ⚠️，请查看 logs/ 下对应日志"
  echo "======================================================================"
fi
echo
echo "说明：model-free（NeRF）路径还需要 BundleSDF 的 CUDA 算子，本项目默认不构建："
echo "      cd framework_src/bundlesdf/mycuda && pip install --no-build-isolation -e ."
exit $rc