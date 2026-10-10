#!/usr/bin/env bash
# 串行完成剩余编译：等待 nvdiffrast -> pytorch3d -> mycpp
set -u
cd /home/jetson/zhangtianqi/FoundationPose_AGXOrin
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="$PWD/venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY=./venv310/bin/python
LOG=logs/build_rest.log
: > "$LOG"

echo "### 1) 等待 nvdiffrast 编译结束 ..." | tee -a "$LOG"
while pgrep -f "no-build-isolation /tmp/nvdiffrast" >/dev/null; do sleep 15; done
echo "### nvdiffrast 状态:" | tee -a "$LOG"
$PY -c "import nvdiffrast;print('nvdiffrast OK')" >> "$LOG" 2>&1
tail -3 logs/build_nvdiffrast2.log | tr '\r' '\n' | tail -3 >> "$LOG" 2>&1

echo "### 2) 编译 pytorch3d ..." | tee -a "$LOG"
$PY -m pip install --no-cache-dir --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable" >> "$LOG" 2>&1
echo "### pytorch3d 状态:" | tee -a "$LOG"
$PY -c "import pytorch3d;print('pytorch3d OK', pytorch3d.__version__)" >> "$LOG" 2>&1

echo "### 3) 编译 mycpp ..." | tee -a "$LOG"
cd framework_src/mycpp
rm -rf build && mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE="$PWD/../../../venv310/bin/python" >> "$LOG" 2>&1
make -j"$(nproc)" >> "$LOG" 2>&1
echo "### mycpp 编译结束 (检查 .so):" | tee -a "$LOG"
find "$(pwd)" -name "*.so" >> "$LOG" 2>&1

echo "### ALL_BUILD_STEPS_DONE" | tee -a "$LOG"