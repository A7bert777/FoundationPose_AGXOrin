"""临时补丁：绕过 libtorch_cuda_linalg.so 对 cusolverDnXsyevBatched 的依赖。

背景：Jetson (CUDA 12.6) 自带的 cuSOLVER 为 11.6.4.69，缺少 torch 2.8.0 需要的
cusolverDnXsyevBatched 等新符号；而 pip 的 nvidia-cusolver-cu12>=11.7 在 Tegra
上无法创建句柄。因此在 CUDA 上调用 torch.linalg.inv 会因 .so 动态加载失败而报错。

方案：FoundationPose 中的矩阵求逆都是 3x3 / 4x4 等小矩阵（相机、变换），
对它们改在 CPU 上求逆再搬回 CUDA，速度开销可忽略，且不触发故障的 linalg .so。
"""
import torch

_MAX = 16  # 小矩阵维度阈值


def _is_small(t):
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dim() >= 2
        and t.shape[-1] <= _MAX
        and t.shape[-2] <= _MAX
    )


_orig_linalg_inv = torch.linalg.inv


def _linalg_inv(A, *args, **kwargs):
    if _is_small(A) and not (args or kwargs):
        return _orig_linalg_inv(A.cpu()).to(A.device)
    return _orig_linalg_inv(A, *args, **kwargs)


torch.linalg.inv = _linalg_inv

_Tensor = torch.Tensor
_orig_tensor_inverse = _Tensor.inverse


def _tensor_inverse(self, *args, **kwargs):
    if _is_small(self) and not (args or kwargs):
        return _orig_tensor_inverse(self.cpu()).to(self.device)
    return _orig_tensor_inverse(self, *args, **kwargs)


_Tensor.inverse = _tensor_inverse

if hasattr(torch, "inverse"):
    _orig_inverse = torch.inverse

    def _torch_inverse(A, *args, **kwargs):
        if _is_small(A) and not (args or kwargs):
            return _orig_inverse(A.cpu()).to(A.device)
        return _orig_inverse(A, *args, **kwargs)

    torch.inverse = _torch_inverse

# torch.linalg.solve / lstsq 等同样可能触发故障的 linalg 库，做同样的降级
_orig_solve = torch.linalg.solve


def _linalg_solve(A, B, *args, **kwargs):
    if _is_small(A) and isinstance(B, torch.Tensor) and B.is_cuda and A.device == B.device:
        return _orig_solve(A.cpu(), B.cpu(), *args, **kwargs).to(B.device)
    return _orig_solve(A, B, *args, **kwargs)


torch.linalg.solve = _linalg_solve