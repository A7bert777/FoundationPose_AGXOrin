#!/usr/bin/env python3
"""FoundationPose 用 TensorRT engine 推理（可选路径，命令行与 fp_infer.py 兼容）。

原理（monkey-patch，不修改官方源码）：
    Scorer/Refiner 的**网络前向**（A,B -> 输出）替换为 TRT engine 调用，
    而 nvdiffrast **渲染**仍走 PyTorch。由于 TRT 不支持渲染图，这里只接管
    `model.forward`，输入输出张量与 PyTorch 完全对齐，因此上层无需改动。

用法与 run.sh 一致：
    ./sh/run_trt.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/....obj

可用 --scorer-backend / --refiner-backend 单独切回 pytorch 做 A/B 对比。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "framework_src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "py"))

_T_LAUNCH = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trt_runner import TRTEngine  # noqa: E402
import fp_infer as F  # noqa: E402


class TRTNetWrapper:
    """把一个 torch 网络的前向替换为 TRT engine。

    约定网络 forward(A, B) -> dict，这里把 dict 的键映射到 engine 的输出名。
    """

    def __init__(self, torch_model, engine: TRTEngine, out_key_map: dict):
        self.torch_model = torch_model
        self.engine = engine
        self.out_key_map = out_key_map          # {torch输出键: engine输出名}
        self._orig_forward = torch_model.forward

    def __call__(self, A, B, **kw):
        res = self.engine({"A": A.float().contiguous(), "B": B.float().contiguous()})
        return {k: res[v] for k, v in self.out_key_map.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FoundationPose TensorRT 推理",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 与 fp_infer 完全一致的业务参数
    p.add_argument("--video", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--pose-dir", default=None)
    p.add_argument("--est-refine-iter", type=int, default=5)
    p.add_argument("--track-refine-iter", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--est-refine-only", action="store_true")
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--debug-dir", default=None)
    p.add_argument("--symmetry", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tf32", action="store_true")
    # TRT 专属
    p.add_argument("--scorer-backend", choices=["engine", "pytorch"], default="engine")
    p.add_argument("--refiner-backend", choices=["engine", "pytorch"], default="engine")
    return p.parse_args(argv)


def install_trt(cache, use_scorer, use_refiner):
    sb = ROOT / "models" / "scorer.engine"
    rb = ROOT / "models" / "refiner.engine"
    info = []
    if use_scorer and sb.is_file():
        eng = TRTEngine(sb)
        cache["scorer"].model.forward = TRTNetWrapper(
            cache["scorer"].model, eng, {"score_logit": "score_logit"}).__call__
        info.append(f"scorer<-{sb.name}")
    if use_refiner and rb.is_file():
        eng = TRTEngine(rb)
        cache["refiner"].model.forward = TRTNetWrapper(
            cache["refiner"].model, eng, {"trans": "trans", "rot": "rot"}).__call__
        info.append(f"refiner<-{rb.name}")
    return info


def build_estimator_trt(args, cache=None):
    cache = cache if cache is not None else {}
    est, mesh, cache = F.build_estimator(args, cache=cache)
    info = install_trt(cache,
                       args.scorer_backend == "engine",
                       args.refiner_backend == "engine")
    print(f"[TRT] 已接管: {', '.join(info) if info else '无（回退 PyTorch）'}")
    return est, mesh, cache


def main(argv=None):
    args = parse_args(argv)
    # 复用 fp_infer 的校验与结果汇总
    if not Path(args.video).is_dir():
        sys.exit(f"[错误] 找不到场景目录: {args.video}")
    if not Path(args.mesh).is_file():
        sys.exit(f"[错误] 找不到网格文件: {args.mesh}")

    F.set_seed_all(0)
    t0 = time.time()
    est, mesh, cache = build_estimator_trt(args)
    print(f"[计时] 初始化(含 TRT 接管) {time.time() - t0:.1f}s")

    summary = F.run_scene(est, mesh, args, cache=cache, warmup=False, announce=True)
    F.report_bench(summary, args)
    print(f"[计时] 端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())