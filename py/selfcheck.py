#!/usr/bin/env python3
"""环境自检：确认 FoundationPose 在 Jetson 上可用。

检查项：
  1. 关键模块可导入（torch / pytorch3d / nvdiffrast / trimesh / open3d / warp / kornia）
  2. CUDA 可用、设备名称、显存
  3. mycpp 原生扩展可导入
  4. 官方源码 estimater 可导入
  5. Scorer / Refiner 权重与示例数据存在
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "framework_src"
sys.path.insert(0, str(SRC))

ok = True


def check(desc: str, passed: bool, detail: str = "") -> None:
    global ok
    mark = "✅" if passed else "❌"
    print(f"  {mark} {desc}" + (f"  —— {detail}" if detail else ""))
    if not passed:
        ok = False


print("=" * 50)
print(" FoundationPose_AGXOrin 环境自检")
print("=" * 50)

# 1. 模块导入
try:
    import torch
    check("导入 torch", True, f"torch={torch.__version__}")
except Exception as e:  # noqa: BLE001
    check("导入 torch", False, repr(e))
    sys.exit(1)

for mod in ("torchvision", "pytorch3d", "nvdiffrast.torch", "trimesh",
            "open3d", "cv2", "kornia", "warp"):
    try:
        __import__(mod)
        check(f"导入 {mod}", True)
    except Exception as e:  # noqa: BLE001
        check(f"导入 {mod}", False, repr(e))

# 2. CUDA
try:
    cuda = torch.cuda.is_available()
    detail = torch.cuda.get_device_name(0) if cuda else "不可用"
    if cuda:
        free, total = torch.cuda.mem_get_info()
        detail += f" | 显存 {total / 2**30:.1f} GiB（空闲 {free / 2**30:.1f} GiB）"
    check("CUDA 可用", cuda, detail)
except Exception as e:  # noqa: BLE001
    check("CUDA 可用", False, repr(e))

# 2b. warp 自己的 CUDA 设备表
# warp-lang 内嵌一份 CUDA Toolkit，若其版本高于本机驱动可支持的上限，
# warp **不会报错**，而是静默退化为纯 CPU 模式（cuda_devices 为空）。
# 之后 framework_src/Utils.py 的 erode_depth() 调 wp.from_torch(..., 'cuda')
# 会抛一个离病根很远的 IndexError: list index out of range，极难排查。
# 这里提前把真正的原因点出来。
try:
    import warp as wp
    wp.init()
    n_cuda = len(wp.get_cuda_devices())
    if n_cuda > 0:
        check("warp 识别到 CUDA 设备", True, f"warp={wp.config.version}")
    else:
        check("warp 识别到 CUDA 设备", False,
              f"warp={wp.config.version} 未注册任何 CUDA 设备（已退化为纯 CPU 模式）。"
              "原因通常是 warp-lang 版本过新、其内嵌 CUDA Toolkit 高于本机驱动。"
              "修复：pip install 'warp-lang==1.16.0'")
except Exception as e:  # noqa: BLE001
    check("warp 识别到 CUDA 设备", False, repr(e))

# 3. mycpp
build_dir = SRC / "mycpp" / "build"
so_files = list(build_dir.glob("mycpp*.so"))
check("mycpp 原生扩展存在", bool(so_files),
      so_files[0].name if so_files else f"未找到（{build_dir}），请运行 sh/build_ext.sh mycpp")
if so_files:
    try:
        sys.path.insert(0, str(build_dir))
        import mycpp  # noqa: F401
        check("导入 mycpp", True)
    except Exception as e:  # noqa: BLE001
        check("导入 mycpp", False, repr(e))

# 4. estimater（官方源码）
try:
    import estimater  # noqa: F401
    check("导入 estimater（官方源码）", True)
except Exception as e:  # noqa: BLE001
    check("导入 estimater（官方源码）", False, repr(e))

# 5. 资源
scorer = ROOT / "weights" / "2024-01-11-20-02-45"
refiner = ROOT / "weights" / "2023-10-28-18-33-37"
check("Scorer 权重存在", (scorer / "model_best.pth").is_file() and (scorer / "config.yml").is_file(),
      str(scorer) if (scorer / "model_best.pth").is_file() else "缺失，请执行 ./sh/download.sh")
check("Refiner 权重存在", (refiner / "model_best.pth").is_file() and (refiner / "config.yml").is_file(),
      str(refiner) if (refiner / "model_best.pth").is_file() else "缺失，请执行 ./sh/download.sh")
demo = ROOT / "demo_data" / "mustard0" / "mesh" / "textured_simple.obj"
check("示例数据 demo_data/mustard0", demo.is_file(),
      "ok" if demo.is_file() else "缺失，请执行 ./sh/download.sh")

print("=" * 50)
if ok:
    print(" 结论：环境就绪，可以开始位姿估计与跟踪。")
else:
    print(" 结论：存在未通过项，请按上述提示修复。")
sys.exit(0 if ok else 1)