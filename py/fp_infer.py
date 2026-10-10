#!/usr/bin/env python3
"""FoundationPose 6D 位姿估计与跟踪推理（AGX Orin）。

官方用法（run_demo.py）::

    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    mesh = trimesh.load(mesh_file)
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    # 首帧：注册（给出目标掩码）
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
    # 后续帧：跟踪
    pose = est.track_one(rgb=color, depth=depth, K=K, iteration=2)

用法::

    ./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj \\
                --out outputimage/mustard_track.png
    ./sh/run.sh --video <scene_dir> --mesh <obj> --est-refine-iter 5 --track-refine-iter 2

本模块同时被 :mod:`fp_server`（常驻服务）复用：服务的每次请求都会调用
:func:`prepare_args` + :func:`run_scene`，因此「命令行单次运行」与「常驻服务推理」
走的是**同一套代码**，输出完全一致。

关于输入场景目录
----------------
沿用官方 ``YcbineoatReader`` 约定：目录下需含
  - ``rgb/*.png``（或 ``color/``）    彩色帧
  - ``depth/*.png``（或 ``depth/``）  深度帧（uint16，单位 mm）
  - ``masks/*.png``（或 ``masks/``）  首帧目标掩码（仅首帧必需）
本模块据此自动解析，缺省与官方 datareader 一致。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "framework_src"
# 让 `import estimater` / `import datareader` 能找到官方源码
sys.path.insert(0, str(SRC))

_T_LAUNCH = time.time()          # 进程启动时刻（初始化耗时对比基准）

import numpy as np
import torch

# Jetson(CUDA12.6) 自带 cuSOLVER 11.6 缺 torch2.8 需要的 cusolverDnXsyevBatched，
# 用 CPU 降级小矩阵求逆，绕过 libtorch_cuda_linalg.so 加载失败（详见补丁文件说明）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fp_linalg_patch  # noqa: F401,E402

_T_IMPORT = time.time() - _T_LAUNCH   # numpy/torch 导入完成耗时


# --------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FoundationPose 6D 位姿估计与跟踪（AGX Orin）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", required=True, help="场景目录（含 rgb/depth/masks）")
    p.add_argument("--mesh", required=True, help="物体 CAD 网格（.obj/.ply）")
    p.add_argument("--out", default=None, help="输出可视化图像（最后一帧）")
    p.add_argument("--out-video", default=None, help="输出可视化视频（.mp4）")
    p.add_argument("--pose-dir", default=None, help="把每帧位姿保存为 txt 到此目录")
    p.add_argument("--est-refine-iter", type=int, default=5, help="注册阶段精修迭代次数")
    p.add_argument("--track-refine-iter", type=int, default=2, help="跟踪阶段精修迭代次数")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理帧数（0=全部）")
    p.add_argument("--est-refine-only", action="store_true",
                   help="只做单帧注册（不做后续跟踪）")
    p.add_argument("--debug", type=int, default=0, help="调试级别（>=1 绘制位姿轴）")
    p.add_argument("--debug-dir", default=None, help="调试输出目录")
    p.add_argument("--symmetry", action="store_true",
                   help="从场景 info.yml 读取对称变换（若存在）")
    p.add_argument("--bench", action="store_true", help="打印耗时基准")
    p.add_argument("--repeat", type=int, default=3, help="基准重复次数（针对注册帧）")
    p.add_argument("--device", default="cuda", help="cuda / cpu（cpu 仅用于排查）")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    return p.parse_args(argv)


def prepare_args(argv=None):
    """解析命令行并做基本校验（本地运行与常驻服务共用）。

    失败时抛 ``SystemExit``（携带中文错误信息）。
    """
    args = parse_args(argv)
    if not Path(args.video).is_dir():
        sys.exit(f"[错误] 找不到场景目录: {args.video}")
    if not Path(args.mesh).is_file():
        sys.exit(f"[错误] 找不到网格文件: {args.mesh}")
    return args


# --------------------------------------------------------------------- #
# 场景读取（对齐官方 YcbineoatReader 的文件约定）
# --------------------------------------------------------------------- #
def _find_subdir(scene: Path, candidates) -> Path | None:
    for name in candidates:
        d = scene / name
        if d.is_dir():
            return d
    return None


def _sorted_files(d: Path) -> list[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)


DEFAULT_K = np.array([[572.4114, 0.0, 325.2611],
                      [0.0, 573.5704, 242.0490],
                      [0.0, 0.0, 1.0]], dtype=np.float64)


def load_intrinsics(scene: Path) -> np.ndarray:
    """读取相机内参：优先 scene/cam_K.txt，否则用默认（LINEMOD 风格）。"""
    for name in ("cam_K.txt", "K.txt", "intrinsics.txt"):
        f = scene / name
        if f.is_file():
            K = np.loadtxt(f).reshape(3, 3)
            print(f"[信息] 使用内参 {f.name}")
            return K
    print(f"[信息] 未找到内参文件，使用默认 K（如结果异常请提供 {scene}/cam_K.txt）")
    return DEFAULT_K.copy()


class SceneReader:
    """极简场景读取器：rgb/depth/masks 三序列 + 内参。"""

    def __init__(self, scene_dir: str):
        self.scene = Path(scene_dir)
        self.rgb_dir = _find_subdir(self.scene, ("rgb", "color", "color_images"))
        self.depth_dir = _find_subdir(self.scene, ("depth", "depths"))
        self.mask_dir = _find_subdir(self.scene, ("masks", "mask", "ob_mask"))
        if self.rgb_dir is None or self.depth_dir is None:
            sys.exit(f"[错误] 场景 {self.scene} 需包含 rgb/ 与 depth/ 子目录")
        self.color_files = _sorted_files(self.rgb_dir)
        self.depth_files = _sorted_files(self.depth_dir)
        self.mask_files = _sorted_files(self.mask_dir) if self.mask_dir else []
        if not self.color_files:
            sys.exit(f"[错误] {self.rgb_dir} 下没有图像")
        self.K = load_intrinsics(self.scene)
        self.H = self.W = None

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i: int) -> np.ndarray:
        import imageio.v2 as imageio
        img = imageio.imread(self.color_files[i])
        if img.ndim == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        return img[..., :3].astype(np.uint8)

    def get_depth(self, i: int) -> np.ndarray:
        import imageio.v2 as imageio
        d = imageio.imread(self.depth_files[i])
        # 官方约定：深度以 mm 存储的 uint16；若已是米（float）则不缩放
        if d.dtype == np.uint16:
            return d.astype(np.float32) / 1000.0
        return d.astype(np.float32)

    def get_mask(self, i: int) -> np.ndarray:
        if not self.mask_files:
            sys.exit(f"[错误] 未提供掩码（{self.scene}/masks/），首帧注册需要目标掩码")
        import imageio.v2 as imageio
        m = imageio.imread(self.mask_files[min(i, len(self.mask_files) - 1)])
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)


# --------------------------------------------------------------------- #
# 模型构建
# --------------------------------------------------------------------- #
def check_weights():
    """确认 Scorer / Refiner 权重存在。"""
    need = {
        "Scorer": ROOT / "weights" / "2024-01-11-20-02-45" / "model_best.pth",
        "Refiner": ROOT / "weights" / "2023-10-28-18-33-37" / "model_best.pth",
    }
    missing = [f"{k}({v})" for k, v in need.items() if not v.is_file()]
    if missing:
        sys.exit(f"[错误] 缺少权重: {', '.join(missing)}，请先执行 ./sh/download.sh")


def set_seed_all(seed=0):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_estimator(args, cache: dict | None = None):
    """构建 ScorePredictor / PoseRefinePredictor / glctx / FoundationPose。

    cache 允许常驻服务复用已加载的网络与 glctx（它们与物体无关）。
    """
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    check_weights()
    if not args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache = cache if cache is not None else {}

    t0 = time.time()
    if "scorer" not in cache:
        cache["scorer"] = ScorePredictor()
    if "refiner" not in cache:
        cache["refiner"] = PoseRefinePredictor()
    if "glctx" not in cache:
        cache["glctx"] = dr.RasterizeCudaContext()
    print(f"[信息] 网络加载 {time.time() - t0:.1f}s")

    t0 = time.time()
    mesh = trimesh.load(args.mesh)
    print(f"[信息] 网格加载 {time.time() - t0:.1f}s | 顶点 {len(mesh.vertices)} 面 {len(mesh.faces)}")

    symmetry_tfs = None
    if args.symmetry:
        info_file = Path(args.video) / "info.yml"
        if info_file.is_file():
            from Utils import symmetry_tfs_from_info
            import ruamel.yaml
            y = ruamel.yaml.YAML()
            with open(info_file) as ff:
                info = y.load(ff)
            symmetry_tfs = symmetry_tfs_from_info(info)
            print(f"[信息] 载入对称变换 {len(symmetry_tfs)} 个")

    debug_dir = args.debug_dir or str(ROOT / "outputs" / "debug")
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        symmetry_tfs=symmetry_tfs, mesh=mesh,
        scorer=cache["scorer"], refiner=cache["refiner"],
        glctx=cache["glctx"], debug=args.debug, debug_dir=debug_dir)
    return est, mesh, cache


# --------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------- #
def draw_pose(frame: np.ndarray, K: np.ndarray, pose: np.ndarray, mesh, to_origin) -> np.ndarray:
    """在帧上绘制位姿 3D 框 + 坐标轴。"""
    import trimesh
    from Utils import draw_posed_3d_box, draw_xyz_axis
    bbox = np.stack([-trimesh.bounds.oriented_bounds(mesh)[1] / 2,
                     trimesh.bounds.oriented_bounds(mesh)[1] / 2], axis=0).reshape(2, 3)
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = frame[..., ::-1].copy() if frame.shape[2] == 3 else frame.copy()
    vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K,
                        thickness=3, transparency=0, is_input_rgb=False)
    return vis[..., ::-1]


# --------------------------------------------------------------------- #
# 推理主流程
# --------------------------------------------------------------------- #
def run_scene(est, mesh, args, cache: dict | None = None, warmup=False, announce=True) -> dict:
    """读场景 -> 首帧注册 -> 逐帧跟踪 -> 保存，返回汇总 dict。

    本地运行与常驻服务共用此入口。
    """
    import trimesh

    reader = SceneReader(args.video)
    n = len(reader)
    if args.max_frames and args.max_frames > 0:
        n = min(n, args.max_frames)
    if announce:
        print(f"[信息] 场景 {args.video} | 共 {len(reader)} 帧，处理 {n} 帧")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    pose_dir = Path(args.pose_dir) if args.pose_dir else None
    if pose_dir:
        pose_dir.mkdir(parents=True, exist_ok=True)

    poses = []
    last_vis = None
    times = []

    for i in range(n):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

        t0 = time.time()
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=args.est_refine_iter)
            stage = "register"
        else:
            if args.est_refine_only:
                break
            pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                 iteration=args.track_refine_iter)
            stage = "track"
        sync()
        dt = time.time() - t0
        times.append((stage, dt))

        poses.append(pose)
        if announce:
            print(f"       [{i+1}/{n}] {stage:8s} {dt*1000:7.1f}ms  "
                  f"t=({pose[0,3]:+.3f},{pose[1,3]:+.3f},{pose[2,3]:+.3f})")

        if pose_dir:
            np.savetxt(pose_dir / f"{i:06d}.txt", pose.reshape(4, 4))

        if args.debug >= 1 or (i == n - 1 and args.out):
            last_vis = draw_pose(color, reader.K, pose, mesh, to_origin)

    # 保存
    if args.out and last_vis is not None:
        import cv2
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), last_vis[..., ::-1])
        print(f"[输出] 可视化 -> {out_path}")

    reg_t = [t for s, t in times if s == "register"]
    trk_t = [t for s, t in times if s == "track"]
    summary = {
        "frames": n,
        "register_ms": reg_t[0] * 1000 if reg_t else None,
        "track_ms_mean": (float(np.mean(trk_t)) * 1000) if trk_t else None,
        "times": times,
        "pose_last": poses[-1] if poses else None,
    }
    return summary


def report_bench(summary, args):
    a = np.array([t for _, t in summary["times"]]) * 1000
    print("\n[汇总] 耗时统计（ms）:")
    if summary["register_ms"] is not None:
        print(f"       注册帧: {summary['register_ms']:.1f}ms（精修迭代 {args.est_refine_iter}）")
    if summary["track_ms_mean"] is not None:
        print(f"       跟踪帧: 均值 {summary['track_ms_mean']:.1f}ms"
              f"（精修迭代 {args.track_refine_iter}）")
    if len(a):
        print(f"       端到端: 均值 {a.mean():.1f}ms | 中位 {np.median(a):.1f}ms | "
              f"最小 {a.min():.1f}ms | 最大 {a.max():.1f}ms")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"[显存] 分配 {torch.cuda.memory_allocated()/2**30:.2f}GiB | "
              f"峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GiB")


def main(argv=None):
    args = prepare_args(argv)
    set_seed_all(0)

    t_build = time.time()
    est, mesh, cache = build_estimator(args)
    t_model = time.time() - t_build
    print(f"[计时] 初始化: 导入三方库 {_T_IMPORT:.1f}s | 建估计器 {t_model:.1f}s | "
          f"初始化总计 {time.time() - _T_LAUNCH:.1f}s")

    if args.bench:
        # 针对注册帧做重复基准（建模开销只在首帧）
        reader = SceneReader(args.video)
        color, depth = reader.get_color(0), reader.get_depth(0)
        mask = reader.get_mask(0).astype(bool)
        print(f"\n[基准] 注册帧 重复{args.repeat}次（est_refine_iter={args.est_refine_iter}）")
        est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)  # 预热
        ts = []
        for i in range(args.repeat):
            t0 = time.time()
            est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            if args.device == "cuda":
                torch.cuda.synchronize()
            ts.append((time.time() - t0) * 1000)
            print(f"  #{i+1}: {ts[-1]:.1f}ms")
        a = np.array(ts)
        print(f"[汇总] 注册: 均值 {a.mean():.1f}ms | 中位 {np.median(a):.1f}ms | "
              f"最小 {a.min():.1f}ms | 最大 {a.max():.1f}ms")
        if args.device == "cuda" and torch.cuda.is_available():
            print(f"[显存] 分配 {torch.cuda.memory_allocated()/2**30:.2f}GiB | "
                  f"峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
        return 0

    summary = run_scene(est, mesh, args, cache, warmup=True, announce=True)
    report_bench(summary, args)
    print(f"[计时] 端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())