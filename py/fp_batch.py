#!/usr/bin/env python3
"""FoundationPose 批量多场景处理（复用 fp_infer 的核心逻辑）。

对一组场景目录依次执行「首帧注册 + 逐帧跟踪」，可统一输出可视化图、每场景位姿
目录，并把每个场景的摘要（帧数、注册/跟踪耗时、末帧位姿）写入 CSV。

Scorer / Refiner / glctx 与物体无关，本脚本**只加载一次**并在所有场景间复用，
因此批量处理比逐个调用 ``sh/run.sh`` 更快。

用法::

    ./venv310/bin/python py/fp_batch.py \
        --input scene_a scene_b \
        --mesh m.obj \
        --outdir outputimage/batch \
        --csv outputs/batch.csv

    # 每场景位姿输出到 <pose-dir>/<场景名>/000000.txt ...
    ./venv310/bin/python py/fp_batch.py --input scene_a --mesh m.obj \
        --pose-dir outputs/batch_poses
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "framework_src"))

_T_LAUNCH = time.time()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FoundationPose 批量多场景处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="一个或多个场景目录")
    p.add_argument("--mesh", required=True, help="物体 CAD 网格（.obj/.ply）")
    p.add_argument("--outdir", default=None, help="可视化输出目录（每场景一张 PNG）")
    p.add_argument("--pose-dir", default=None,
                   help="位姿输出根目录（每场景一个子目录）")
    p.add_argument("--csv", default=None, help="把每场景摘要写入 CSV")
    p.add_argument("--est-refine-iter", type=int, default=5)
    p.add_argument("--track-refine-iter", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--est-refine-only", action="store_true")
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--symmetry", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tf32", action="store_true")
    return p.parse_args(argv)


def scene_name(scene: str) -> str:
    """从场景路径提取一个可用的名字（末级目录名）。"""
    return Path(scene.rstrip("/")).name or "scene"


def build_scene_ns(base, scene: str, out_png, pose_dir):
    """为单个场景构造与 fp_infer 兼容的参数命名空间。"""
    return argparse.Namespace(
        video=scene,
        mesh=base.mesh,
        out=out_png,
        out_video=None,
        pose_dir=pose_dir,
        est_refine_iter=base.est_refine_iter,
        track_refine_iter=base.track_refine_iter,
        max_frames=base.max_frames,
        est_refine_only=base.est_refine_only,
        debug=base.debug,
        debug_dir=None,
        symmetry=base.symmetry,
        bench=False,
        repeat=3,
        device=base.device,
        no_tf32=base.no_tf32,
    )


def _fmt(v):
    """把数值格式化为保留两位小数（None 原样返回）。"""
    if v is None:
        return None
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return v


def main(argv=None) -> int:
    base = parse_args(argv)

    if not Path(base.mesh).is_file():
        sys.exit(f"[错误] 找不到网格文件: {base.mesh}")
    for s in base.input:
        if not Path(s).is_dir():
            sys.exit(f"[错误] 找不到场景目录: {s}")

    import fp_infer as F

    F.set_seed_all(0)
    cache: dict = {}          # 跨场景复用 scorer / refiner / glctx

    if base.outdir:
        Path(base.outdir).mkdir(parents=True, exist_ok=True)
    if base.pose_dir:
        Path(base.pose_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.time()
    for idx, scene in enumerate(base.input, 1):
        name = scene_name(scene)
        print(f"\n[批量] ({idx}/{len(base.input)}) 场景 {name} <- {scene}")

        out_png = str(Path(base.outdir) / f"{name}.png") if base.outdir else None
        pose_dir = str(Path(base.pose_dir) / name) if base.pose_dir else None

        ns = build_scene_ns(base, scene, out_png, pose_dir)
        t0 = time.time()
        try:
            est, mesh, cache = F.build_estimator(ns, cache=cache)
            summary = F.run_scene(est, mesh, ns, cache=cache,
                                  warmup=True, announce=True)
            dt = time.time() - t0
            pose_last = summary.get("pose_last")
            t = pose_last[:3, 3] if pose_last is not None else None
            rows.append({
                "scene": name,
                "path": scene,
                "frames": summary.get("frames"),
                "register_ms": _fmt(summary.get("register_ms")),
                "track_ms_mean": _fmt(summary.get("track_ms_mean")),
                "total_s": round(dt, 2),
                "tx": _fmt(t[0] if t is not None else None),
                "ty": _fmt(t[1] if t is not None else None),
                "tz": _fmt(t[2] if t is not None else None),
                "status": "ok",
            })
        except SystemExit as e:
            print(f"[批量] 场景 {name} 跳过：{e}")
            rows.append({"scene": name, "path": scene, "frames": None,
                         "register_ms": None, "track_ms_mean": None,
                         "total_s": None, "tx": None, "ty": None, "tz": None,
                         "status": f"skip: {e}"})
        except Exception as e:  # noqa: BLE001
            print(f"[批量] 场景 {name} 失败：{e!r}")
            rows.append({"scene": name, "path": scene, "frames": None,
                         "register_ms": None, "track_ms_mean": None,
                         "total_s": None, "tx": None, "ty": None, "tz": None,
                         "status": f"error: {e!r}"})

    print("\n[批量] 汇总:")
    for r in rows:
        print(f"  {r['scene']:20s} frames={r['frames']} "
              f"reg={r['register_ms']}ms trk={r['track_ms_mean']}ms "
              f"总={r['total_s']}s  {r['status']}")

    if base.csv:
        Path(base.csv).parent.mkdir(parents=True, exist_ok=True)
        fields = ["scene", "path", "frames", "register_ms", "track_ms_mean",
                  "total_s", "tx", "ty", "tz", "status"]
        with open(base.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[批量] 摘要已写入 {base.csv}")

    print(f"[计时] 批量端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s "
          f"| 处理 {len(base.input)} 个场景，用时 {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())