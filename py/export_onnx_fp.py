#!/usr/bin/env python3
"""把 FoundationPose 的 Scorer / Refiner 网络导出为 ONNX（FP32）。

背景与取舍
----------
FoundationPose 的推理 = 网络前向 + nvdiffrast 可微渲染 + 迭代精修。
其中**渲染**必须留在 PyTorch/nvdiffrast（TRT 不支持该图），能 TRT 化的只有
Scorer/Refiner 的卷积网络主干。因此本路径为**可选优化**：
  * Scorer  : 输入一对裁剪图 (A, B)，输出 score_logit
  * Refiner : 输入一对裁剪图 (A, B)，输出 trans / rot 增量

本脚本导出这两个网络为 ONNX，供 build_engine_fp.sh 构建 engine。

用法：
    ./venv310/bin/python py/export_onnx_fp.py                 # 两个都导出
    ./venv310/bin/python py/export_onnx_fp.py --part scorer
    ./venv310/bin/python py/export_onnx_fp.py --part refiner
    ./venv310/bin/python py/export_onnx_fp.py --no-verify      # 跳过数值校验
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "framework_src"
sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn


class RefinerWrapper(nn.Module):
    """只包装 RefineNet 的前向：A,B -> trans / rot。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        out = self.model(A, B)
        return out["trans"], out["rot"]


class ScorerWrapper(nn.Module):
    """只包装 ScoreNetMultiPair 的前向：A,B -> score_logit。

    注意 ScorePredictor 内部有 pair 竞赛逻辑；此处导出的是**单次前向**，
    竞赛在 Python 侧完成。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        out = self.model(A, B, L=1)
        return out["score_logit"]


def load_predictor(kind: str, device: str):
    from estimater import ScorePredictor, PoseRefinePredictor
    if kind == "scorer":
        p = ScorePredictor()
        return p.model, p.cfg
    p = PoseRefinePredictor()
    return p.model, p.cfg


def export_one(wrapper, args_io, out_path: Path, opset, input_names, output_names,
               dynamic_axes=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [导出] {out_path.name} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, args_io, str(out_path),
            input_names=input_names, output_names=output_names,
            opset_version=opset, do_constant_folding=True, export_params=True,
            dynamic_axes=dynamic_axes, dynamo=False)
    dt = time.time() - t0
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"  [导出] 完成 {dt:.1f}s | 大小 {size_mb:.1f} MB")
    return dt


def check_onnx(out_path: Path):
    try:
        import onnx
    except ImportError:
        print("     (未安装 onnx，跳过结构校验)")
        return
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    print(f"     ✅ ONNX 校验通过 | 节点数 {len(m.graph.node)}")


def verify(out_path: Path, wrapper, ref, dummy, out_names):
    try:
        import onnxruntime as ort
    except ImportError:
        print("     (未安装 onnxruntime，跳过数值校验)")
        return
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: dummy[0].detach().cpu().numpy(),
            sess.get_inputs()[1].name: dummy[1].detach().cpu().numpy()}
    got = sess.run(None, feed)
    worst = 0.0
    for name, r, g in zip(out_names, ref, got):
        rn = r.detach().cpu().numpy()
        diff = float(abs(rn - g).max())
        worst = max(worst, diff)
        print(f"     {name:<12} max|Δ|={diff:.3e}")
    print(f"     最大绝对误差 = {worst:.3e} → {'✅ 可忽略' if worst < 1e-3 else '⚠️ 请检查'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="FoundationPose Scorer/Refiner -> ONNX (FP32)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", default=str(ROOT / "models"))
    p.add_argument("--res", type=int, default=160, help="输入裁剪分辨率（边长）")
    p.add_argument("--c-in", type=int, default=7, help="输入通道数（rgb3+xyz3[+normal1]）")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--part", choices=["scorer", "refiner", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    print("=" * 72)
    print(" FoundationPose  Scorer / Refiner  →  ONNX   (FP32)")
    print("=" * 72)
    print(f"  输入尺寸 : 2 x {args.c_in} x {args.res} x {args.res}")
    print(f"  输出目录 : {outdir}")
    print(f"  opset    : {args.opset}")
    print("=" * 72)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    results = []
    dummy = (torch.randn(1, args.c_in, args.res, args.res, device=args.device),
             torch.randn(1, args.c_in, args.res, args.res, device=args.device))

    if args.part in ("both", "scorer"):
        print("\n① Scorer (ScoreNetMultiPair → score_logit)")
        model, _ = load_predictor("scorer", args.device)
        wrap = ScorerWrapper(model).to(args.device).eval()
        with torch.no_grad():
            ref = wrap(*dummy)
        ref = (ref if isinstance(ref, (tuple, list)) else (ref,))
        out = outdir / "scorer.onnx"
        export_one(wrap, dummy, out, args.opset, ["A", "B"], ["score_logit"])
        check_onnx(out)
        results.append(("scorer", out, ref, ["score_logit"]))

    if args.part in ("both", "refiner"):
        print("\n② Refiner (RefineNet → trans / rot)")
        model, _ = load_predictor("refiner", args.device)
        wrap = RefinerWrapper(model).to(args.device).eval()
        with torch.no_grad():
            ref = wrap(*dummy)
        out = outdir / "refiner.onnx"
        export_one(wrap, dummy, out, args.opset, ["A", "B"], ["trans", "rot"])
        check_onnx(out)
        results.append(("refiner", out, ref, ["trans", "rot"]))

    if not args.no_verify:
        for kind, out, ref, names in results:
            print(f"\n  [校验] {out.name}")
            if kind == "scorer":
                model, _ = load_predictor("scorer", args.device)
                wrap = ScorerWrapper(model).to(args.device).eval()
            else:
                model, _ = load_predictor("refiner", args.device)
                wrap = RefinerWrapper(model).to(args.device).eval()
            verify(out, wrap, ref, dummy, names)

    print("\n" + "=" * 72)
    print(" 完成。下一步构建 engine：")
    for _, out, _, _ in results:
        print(f"   bash sh/build_engine_fp.sh {out} fp32")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())