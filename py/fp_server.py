#!/usr/bin/env python3
"""FoundationPose 常驻服务（Unix domain socket + 一行 JSON 协议）。

作用（对应 README 第 4 节）：
  * 把**与物体无关、加载昂贵**的资源 —— Scorer / Refiner 网络权重、nvdiffrast 的
    ``RasterizeCudaContext`` —— 在**进程内只加载一次**并常驻；
  * 之后每个请求只重新构建 ``FoundationPose``（与具体 mesh 绑定）并推理，
    从而在多场景 / 多帧连续调用时省去重复加载开销；
  * 推理流程与命令行单次运行**共用** ``py/fp_infer.py`` 的
    ``prepare_args`` / ``build_estimator`` / ``run_scene``，因此输出必然一致。

协议（与 ``py/fp_client.py``、``sh/serve.sh`` 对齐，一行 JSON 请求 / 一行 JSON 响应）::

    {"cmd": "infer", "argv": [...]}
        -> {"ok": bool, "stdout": str, "elapsed_s": float, "error": str|null}
    {"cmd": "ping"}
        -> {"ok": true, "device": str, "uptime_s": float, "loaded": bool}
    {"cmd": "stop"}
        -> {"ok": true}   随后服务退出并释放显存

用法（一般由 ``sh/serve.sh`` 调用，不建议手动执行）::

    python py/fp_server.py --socket run/fp.sock [--idle-timeout 1800] [--warmup] [--no-tf32]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "framework_src"))

_T_START = time.time()          # 进程启动时刻（用于 uptime）
_STOP = False                   # stop 请求 / 信号置位
_CACHE: dict | None = None      # 常驻的 scorer / refiner / glctx（与物体无关）


def _handle_signal(signum, frame):  # noqa: ARG001
    """SIGTERM / SIGINT：请求主循环优雅退出。"""
    global _STOP
    _STOP = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FoundationPose 常驻服务")
    p.add_argument("--socket", default=str(ROOT / "run" / "fp.sock"),
                   help="Unix domain socket 路径")
    p.add_argument("--idle-timeout", type=float, default=0.0,
                   help="空闲多少秒后自动退出并释放显存（0=不超时）")
    p.add_argument("--warmup", action="store_true", help="启动时预热一次 CUDA 上下文")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    return p.parse_args(argv)


# --------------------------------------------------------------------- #
# 常驻资源加载
# --------------------------------------------------------------------- #
def load_cache(no_tf32: bool) -> dict:
    """加载与物体无关的常驻资源（Scorer / Refiner / glctx）。

    这些对象在服务生命周期内复用，正是「省去重复加载」的关键。
    """
    import torch

    if not no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    import fp_infer as F
    from estimater import ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr

    F.check_weights()          # 权重缺失时给出中文提示并退出

    t0 = time.time()
    cache = {
        "scorer": ScorePredictor(),
        "refiner": PoseRefinePredictor(),
        "glctx": dr.RasterizeCudaContext(),
    }
    print(f"[服务] 常驻网络加载完成 {time.time() - t0:.1f}s", flush=True)
    return cache


def do_warmup() -> None:
    """一次轻量 CUDA 运算预热设备上下文（避免首个请求的额外延迟）。"""
    import torch

    if torch.cuda.is_available():
        _ = torch.zeros(8, device="cuda") + 1.0
        torch.cuda.synchronize()
    print("[服务] 预热完成", flush=True)


# --------------------------------------------------------------------- #
# 请求处理
# --------------------------------------------------------------------- #
def handle_infer(argv) -> dict:
    """执行一次推理；捕获 stdout 一并回传，保证客户端看到与本地运行一致的日志。"""
    import traceback

    import fp_infer as F

    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            args = F.prepare_args(argv)
            F.set_seed_all(0)
            est, mesh, cache = F.build_estimator(args, cache=_CACHE)
            summary = F.run_scene(est, mesh, args, cache=cache, warmup=True, announce=True)
            F.report_bench(summary, args)
        return {"ok": True, "stdout": buf.getvalue(),
                "elapsed_s": round(time.time() - t0, 3), "error": None}
    except SystemExit as e:
        msg = str(e) if str(e) else "参数/资源校验失败"
        return {"ok": False, "stdout": buf.getvalue(), "elapsed_s": None, "error": msg}
    except Exception:  # noqa: BLE001
        return {"ok": False, "stdout": buf.getvalue(), "elapsed_s": None,
                "error": traceback.format_exc()}


def _send(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def _ping() -> dict:
    import torch

    dev = "cpu"
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        dev = "unknown"
    return {"ok": True, "device": dev,
            "uptime_s": round(time.time() - _T_START, 1),
            "loaded": _CACHE is not None}


def handle_conn(conn: socket.socket) -> None:
    global _STOP
    f = conn.makefile("rb")
    try:
        line = f.readline()
    except OSError:
        line = b""
    if not line:
        return
    try:
        req = json.loads(line.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        _send(conn, {"ok": False, "error": "JSON 解析失败"})
        return

    cmd = req.get("cmd")
    if cmd == "infer":
        _send(conn, handle_infer(req.get("argv", [])))
    elif cmd == "ping":
        _send(conn, _ping())
    elif cmd == "stop":
        _send(conn, {"ok": True})
        _STOP = True
    else:
        _send(conn, {"ok": False, "error": f"未知命令: {cmd}"})
    finally_close(f)


def finally_close(f) -> None:
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- #
# 服务主循环
# --------------------------------------------------------------------- #
def serve(sock_path: str, idle_timeout: float) -> None:
    global _STOP

    sock_path = Path(sock_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        try:
            sock_path.unlink()
        except OSError:
            pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    srv.settimeout(1.0)            # 定期醒来检查 _STOP / 空闲超时
    try:
        os.chmod(sock_path, 0o600)
    except OSError:
        pass

    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[服务] 监听 {sock_path}（device={dev}，idle_timeout={idle_timeout}s）", flush=True)

    last_active = time.time()
    while not _STOP:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            if idle_timeout > 0 and (time.time() - last_active) > idle_timeout:
                print("[服务] 空闲超时，退出并释放显存", flush=True)
                break
            continue
        except OSError:
            break

        last_active = time.time()
        try:
            handle_conn(conn)
        except Exception as e:  # noqa: BLE001
            print(f"[服务] 请求处理异常: {e}", file=sys.stderr, flush=True)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        srv.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        sock_path.unlink()
    except OSError:
        pass
    print("[服务] 已退出", flush=True)


def main(argv=None) -> int:
    global _CACHE

    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print("[服务] 正在加载常驻网络（首次约十数秒）...", flush=True)
    _CACHE = load_cache(args.no_tf32)
    if args.warmup:
        do_warmup()

    serve(args.socket, args.idle_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())