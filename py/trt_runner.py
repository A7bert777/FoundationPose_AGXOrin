#!/usr/bin/env python3
"""TensorRT 10.x engine 推理封装（通用版）。

关键设计（与 SAM3_AGXOrin 一致）：
  * 用 TRT 10 的 **execute_async_v3 + set_tensor_address**，直接拿 torch CUDA
    张量的 data_ptr 当显存地址 —— **不需要 pycuda / cuda-python**。
  * **专用非默认 stream**：避免 TRT 在默认流上的强制同步开销。
  * **输入/输出 buffer 缓存**：首次调用后复用。

用法：
    eng = TRTEngine("models/refiner.engine")
    print(eng.info())
    out = eng({"A": a, "B": b})       # a/b: torch.Tensor
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def _trt_to_torch_dtype(trt_dtype: trt.DataType):
    if trt_dtype == trt.bool:
        return torch.bool
    if trt_dtype == trt.int32:
        return torch.int32
    if trt_dtype == trt.int64:
        return torch.int64
    return torch.from_numpy(np.empty(0, dtype=trt.nptype(trt_dtype))).dtype


class TRTEngine:
    """封装一个 .engine，提供 ``engine({name: tensor}) -> {name: tensor}``。"""

    def __init__(self, engine_path, device: str = "cuda", verbose: bool = False):
        self.path = Path(engine_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 engine: {self.path}")

        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.device = torch.device(device)

        with open(self.path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败: {self.path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"创建 execution context 失败: {self.path}")

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.stream = torch.cuda.Stream(self.device)
        self._in_cache = {}
        self._out_cache = {}
        self._calls = 0

    def info(self) -> str:
        lines = [f"engine : {self.path.name}"]
        for n in self.input_names:
            lines.append(f"  IN   {n:14s} {tuple(self.engine.get_tensor_shape(n))} "
                         f"{self.engine.get_tensor_dtype(n)}")
        for n in self.output_names:
            lines.append(f"  OUT  {n:14s} {tuple(self.engine.get_tensor_shape(n))} "
                         f"{self.engine.get_tensor_dtype(n)}")
        return "\n".join(lines)

    def __call__(self, feeds: dict) -> dict:
        missing = set(self.input_names) - set(feeds)
        if missing:
            raise KeyError(f"缺少输入 {missing}，需要 {self.input_names}")

        buffers = {}
        shapes_ready = self._calls > 0

        for name in self.input_names:
            src = feeds[name]
            if not isinstance(src, torch.Tensor):
                src = torch.from_numpy(np.asarray(src))
            if src.dtype == torch.float64:
                src = src.float()
            src = src.to(self.device).contiguous()

            key = ("in", name, tuple(src.shape), str(src.dtype))
            buf = self._in_cache.get(key)
            if buf is None:
                buf = torch.empty_like(src)
                self._in_cache[key] = buf
                shapes_ready = False
            buf.copy_(src, non_blocking=True)
            buffers[name] = buf

            if not shapes_ready:
                try:
                    self.context.set_input_shape(name, tuple(src.shape))
                except Exception:
                    pass
            if not self.context.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"set_tensor_address 失败（输入 {name}）")

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"输出 {name} shape 未定 {shape}，输入 shape 未绑定？")
            key = ("out", name, shape)
            buf = self._out_cache.get(key)
            if buf is None:
                buf = torch.empty(shape, dtype=_trt_to_torch_dtype(
                    self.engine.get_tensor_dtype(name)), device=self.device)
                self._out_cache[key] = buf
            buffers[name] = buf
            if not self.context.set_tensor_address(name, buf.data_ptr()):
                raise RuntimeError(f"set_tensor_address 失败（输出 {name}）")

        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"execute_async_v3 失败: {self.path.name}")
        self.stream.synchronize()
        self._calls += 1

        return {n: buffers[n] for n in self.output_names}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python trt_runner.py <engine>")
        sys.exit(1)
    e = TRTEngine(sys.argv[1])
    print(e.info())