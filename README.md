# FoundationPose_AGXOrin

在 **NVIDIA Jetson AGX Orin (64GB)** 上部署 **NVIDIA FoundationPose**（CVPR 2024 Highlight），
面向**单目/单目+深度的 6D 物体位姿估计与跟踪**：给定物体的 CAD 模型（或少量参考图），
无需针对新物体重新训练，即可在**首帧注册**出物体在相机坐标系下的精确位置与朝向，
并对其后的每一帧**持续跟踪** 6D 位姿。

本项目在官方 `NVlabs/FoundationPose` 之上做了三类工程化工作：

1. **Jetson 适配** —— 解决 aarch64 上 pytorch3d / nvdiffrast / mycpp 的原生扩展编译，
   并内置「无 kaolin/无 Triton」的稳健导入链；
2. **统一命令行 + 结果可视化** —— 场景目录 / CAD 网格 / 精修迭代次数统一命令行，
   自动解析 `rgb/depth/masks`，输出位姿、可视化与耗时；
3. **常驻服务加速** —— Scorer/Refiner 网络只加载一次，后续每次请求复用，
   多场景/多帧连续推理显著省去重复加载开销（可选，且不启动也能跑）。

> 目录结构与脚本风格对齐同机上的 **SAM3_AGXOrin**（Meta SAM3 的边缘部署），
> 便于两套流水线（分割 + 位姿）协同：**SAM3 分割出目标 → FoundationPose 估计其 6D 位姿**。

---

## 目录

1. [硬件与环境](#1-硬件与环境)
2. [快速开始](#2-快速开始)
3. [命令与参数](#3-命令与参数)
4. [常驻服务（推荐）](#4-常驻服务推荐)
5. [TensorRT 加速（可选）](#5-tensorrt-加速可选)
6. [输入数据约定](#6-输入数据约定)
7. [项目结构](#7-项目结构)
8. [实现要点](#8-实现要点)
9. [性能实测](#9-性能实测)
10. [常见问题](#10-常见问题)
11. [修复与验证记录](#11-修复与验证记录)
12. [许可](#12-许可)

---

## 1. 硬件与环境

| 项目 | 值 |
|------|-----|
| 设备 | Jetson AGX Orin 64GB（统一内存架构，CPU/GPU 共享） |
| 系统 | Linux 5.15（JetPack 6.x / R36.4.7 / Ubuntu 22.04） |
| Python | 3.10.12（本项目独立的 `venv310/`） |
| PyTorch | 2.8.0（NVIDIA 定制 aarch64 wheel） |
| torchvision | 0.23.0 |
| CUDA | 12.6 |
| TensorRT | 10.3（可选，engine 加速路径） |
| 权重 | `weights/2024-01-11-20-02-45`（Scorer）、`weights/2023-10-28-18-33-37`（Refiner） |

> ⚠️ **本项目的 `venv310/` 是独立创建的**，绝不复用 `SAM3_AGXOrin/venv310` 或
> 其它任何已有项目的虚拟环境。

从零初始化（新机器）：

```bash
cd /home/jetson/zhangtianqi/FoundationPose_AGXOrin
./sh/setup.sh        # 建独立 venv + torch + 依赖 + clone 官方源码 + 编译原生扩展 + 下载权重 + 自检
./sh/build_ext.sh    # 单独编译原生扩展（pytorch3d / nvdiffrast / mycpp）
./sh/download.sh     # 仅重新下载权重/示例数据（断点续传）
./sh/check_env.sh    # 检查 CUDA / 编译器 / 扩展 / 包 可用性
```

环境自检：

```bash
./venv310/bin/python py/selfcheck.py
```

---

## 2. 快速开始

> 所有命令都在项目根目录执行：`cd /home/jetson/zhangtianqi/FoundationPose_AGXOrin`

```bash
# 官方示例：机器人操作芥末瓶，首帧注册 + 后续跟踪
./sh/run.sh --video demo_data/mustard0 \
            --mesh  demo_data/mustard0/mesh/textured_simple.obj \
            --out   outputimage/mustard_track.png

# 保存每帧位姿到目录（000000.txt ... 4x4 矩阵）
./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj \
            --pose-dir outputs/mustard_poses

# 只做首帧注册（不做后续跟踪）
./sh/run.sh --video <scene> --mesh <obj> --est-refine-only

# 调整精修迭代次数（精度 vs 速度）
./sh/run.sh --video <scene> --mesh <obj> --est-refine-iter 5 --track-refine-iter 2

# 性能基准（对注册帧重复测量）
./sh/run.sh --video <scene> --mesh <obj> --bench --repeat 3

# 批量处理多个场景
./venv310/bin/python py/fp_batch.py --input scene_a scene_b --mesh m.obj \
            --outdir outputimage/batch --csv outputs/batch.csv
```

---

## 3. 命令与参数

### 3.1 推理参数（`sh/run.sh` 与 `sh/run_trt.sh` 通用）

| 参数 | 说明 | 默认 |
|------|------|------|
| `--video` | 场景目录（含 `rgb/`、`depth/`、`masks/`） | 必需 |
| `--mesh` | 物体 CAD 网格（`.obj`/`.ply`） | 必需 |
| `--out` | 输出可视化图像（最后一帧） | — |
| `--pose-dir` | 每帧位姿保存为 txt 到此目录 | — |
| `--est-refine-iter` | 注册阶段精修迭代次数 | `5` |
| `--track-refine-iter` | 跟踪阶段精修迭代次数 | `2` |
| `--max-frames` | 最多处理帧数（0=全部） | `0` |
| `--est-refine-only` | 只做首帧注册 | 关 |
| `--debug` | 调试级别（>=1 绘制位姿轴） | `0` |
| `--symmetry` | 从 `info.yml` 读取对称变换 | 关 |
| `--bench` / `--repeat` | 打印耗时基准 / 重复次数 | 关 / `3` |
| `--device` | `cuda` 或 `cpu`（cpu 仅排查） | `cuda` |
| `--no-tf32` | 关闭 TF32（默认开启加速） | 关 |

`run_trt.sh` 额外支持 `--scorer-backend {engine,pytorch}` 与 `--refiner-backend {engine,pytorch}`。

> 查看参数帮助请用：`./venv310/bin/python py/fp_infer.py --help`

### 3.2 服务参数（`sh/serve.sh`）

| 子命令 | 说明 |
|--------|------|
| `start [参数]` | 后台启动服务 |
| `stop` | 停止并释放显存 |
| `restart [参数]` | 重启 |
| `status` | 查看状态与显存占用 |
| `log` | 跟踪日志（`run/fp.log`） |
| `fore [参数]` | 前台启动 |

可透传给服务的参数：`--idle-timeout N`、`--warmup`、`--socket PATH`、`--no-tf32`。

### 3.3 批量参数（`py/fp_batch.py`）

| 参数 | 说明 |
|------|------|
| `--input` | 一个或多个场景目录（必需） |
| `--mesh` | CAD 网格（必需） |
| `--outdir` | 可视化输出目录 |
| `--pose-dir` | 位姿输出根目录（每场景一个子目录） |
| `--csv` | 把每个场景摘要写入 CSV |

---

## 4. 常驻服务（推荐）

### 4.1 为什么需要

单次运行会**加载 Scorer/Refiner 权重与初始化渲染上下文**，这部分与具体物体无关。
常驻服务把它留在内存里，后续每次请求只重新构建 `FoundationPose`（与 mesh 相关）并推理。

### 4.2 用法

```bash
# 启动服务（网络加载一次）
./sh/serve.sh start
./sh/serve.sh start --warmup               # 启动时预热
./sh/serve.sh start --idle-timeout 1800    # 空闲 30 分钟自动退出释放显存

# 之后正常推理 —— 命令完全不变
./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj

# 管理
./sh/serve.sh status      # 查看服务状态 + 显存占用
./sh/serve.sh log         # 跟踪服务日志
./sh/serve.sh stop        # 停止并释放模型与显存
./sh/serve.sh restart
./sh/serve.sh fore        # 前台启动（Ctrl+C 即释放）
```

### 4.3 工作原理

```
./sh/run.sh ...  ->  py/fp_client.py
                        ├─ run/fp.sock 存在且服务在线 -> 转发（网络已常驻）
                        └─ 否则                        -> 本地加载单次运行
```

- **通信**：Unix domain socket（默认 `run/fp.sock`），一行 JSON 请求 / 一行 JSON 响应；
- **协议**：`{"cmd":"infer","argv":[...]}`、`{"cmd":"ping"}`、`{"cmd":"stop"}`；
- **对使用者透明**：`run.sh` 的参数与用法**完全不变**；
- **可选优化**：不启动服务也不会报错，客户端会自动回退到本地运行；
- **同一套代码**：服务端与本地运行都调用 `py/fp_infer.py` 的
  `prepare_args()` / `build_estimator()` / `run_scene()`，因此**输出必然一致**。

环境变量：

| 变量 | 说明 |
|------|------|
| `FP_SOCKET` | 自定义 socket 路径（客户端与服务端须一致） |
| `FP_NO_SERVER` | 设为 `1` 强制本地运行（绕过服务） |

---

## 5. TensorRT 加速（可选）

> FoundationPose 的推理包含 **nvdiffrast 可微渲染 + 迭代精修**，TRT 只能接管
> Scorer/Refiner 的**网络前向**，渲染仍需 PyTorch。收益有限，属**可选优化**。

```bash
# 1) 导出 ONNX（生成 models/scorer.onnx、models/refiner.onnx）
./venv310/bin/python py/export_onnx_fp.py

# 2) 构建 FP32 engine
./sh/build_engine_fp.sh models/scorer.onnx  fp32
./sh/build_engine_fp.sh models/refiner.onnx fp32

# 3) 用 engine 推理（参数与 run.sh 完全一致）
./sh/run_trt.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj

# A/B 对比：单独回退某一侧到 PyTorch
./sh/run_trt.sh --video ... --mesh ... --refiner-backend pytorch
```

**建议**：日常使用直接用 PyTorch 路径（`run.sh`）。TRT 路径在 Orin 上收益与
SAM3 类似——瓶颈在访存与渲染，而非 Tensor Core 矩阵乘。

---

## 6. 输入数据约定

场景目录（对齐官方 `YcbineoatReader`）：

```
scene/
├── cam_K.txt        # 可选：3x3 相机内参；缺省用 LINEMOD 风格默认值
├── rgb/             # 彩色帧 *.png（或 color/）
├── depth/           # 深度帧 *.png（uint16，单位 mm；或 depth/）
├── masks/           # 首帧目标掩码（uint8 0/255，仅首帧必需）
└── info.yml         # 可选：含 symmetries_*，配 --symmetry 使用
```

- 深度为 **uint16 毫米**时脚本自动 `/1000` 转为米；若已是米（float）则原样使用。
- 首帧必须提供目标掩码（`register` 需要）；后续帧无需掩码（`track_one` 自动跟踪）。

---

## 7. 项目结构

```
FoundationPose_AGXOrin/
├── README.md
├── requirements.txt
│
├── sh/                           # Shell 入口（只负责环境变量 + 调度）
│   ├── run.sh                    # ⭐ 主入口（自动选择服务或本地）
│   ├── serve.sh                  # ⭐ 常驻服务管理（start/stop/status/log/fore）
│   ├── run_trt.sh                # TensorRT 推理入口（可选）
│   ├── setup.sh                  # 一键环境初始化
│   ├── build_ext.sh              # 编译原生扩展（pytorch3d/nvdiffrast/mycpp）
│   ├── build_engine_fp.sh        # ONNX -> TensorRT engine（FP32）
│   ├── download.sh               # 下载权重与示例数据（断点续传）
│   ├── _common.sh                # 公共函数：run_logged（输出实时可见 + 落盘）
│   └── check_env.sh              # CUDA/编译器/扩展/包 环境检查
│
├── py/                           # Python 实现
│   ├── fp_infer.py               # 推理核心（注册+跟踪+可视化+基准）
│   ├── fp_client.py              # 客户端：服务优先，自动回退本地
│   ├── fp_server.py              # 常驻服务（Unix socket + JSON 协议）
│   ├── fp_batch.py               # 批量多场景处理
│   ├── fp_trt_infer.py           # TensorRT 版推理（参数同 fp_infer.py）
│   ├── trt_runner.py             # TRT engine 加载与执行封装
│   ├── export_onnx_fp.py         # 导出 Scorer/Refiner ONNX
│   └── selfcheck.py              # 环境自检
│
├── framework_src/                # 官方 FoundationPose 源码（git clone）
│   ├── estimater.py              # FoundationPose / ScorePredictor / PoseRefinePredictor
│   ├── datareader.py, Utils.py   # 数据读取与几何/渲染工具
│   ├── learning/                 # 网络定义与权重加载
│   ├── mycpp/                    # 位姿聚类 C++ 扩展（构建后生成 build/mycpp*.so）
│   └── bundlesdf/                # model-free（NeRF）路径（本项目默认不启用）
│
├── weights/                      # 权重（需下载）
│   ├── 2024-01-11-20-02-45/      # Scorer: model_best.pth + config.yml
│   └── 2023-10-28-18-33-37/      # Refiner: model_best.pth + config.yml
│
├── models/                       # ONNX / engine（可选 TRT 路径，需构建）
├── demo_data/                    # 示例场景（需下载）
├── inputimage/                   # 输入图像（可选）
├── outputimage/                  # 可视化输出
├── outputs/                      # 位姿 / CSV / 调试
├── run/                          # 服务运行时文件（socket / pid / log）
├── logs/                         # 构建与依赖日志
└── venv310/                      # ⭐ 本项目独立虚拟环境
```

---

## 8. 实现要点

### 8.1 使用官方 Python API

推理接口源自官方 `run_demo.py`：

```python
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr

mesh    = trimesh.load(mesh_file)
scorer  = ScorePredictor()
refiner = PoseRefinePredictor()
glctx   = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                     mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)

pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)  # 首帧
pose = est.track_one(rgb=color, depth=depth, K=K, iteration=2)               # 后续帧
```

**关键：Scorer/Refiner/glctx 与物体无关**，可常驻复用（服务化正是利用这一点），
只有 `FoundationPose` 实例与具体 mesh 绑定。

### 8.2 原生扩展

| 扩展 | 作用 | 构建方式 |
|------|------|---------|
| **pytorch3d** | 3D 变换、渲染、SDF 等基础算子 | `pip install --no-build-isolation git+.../pytorch3d.git` |
| **nvdiffrast** | 可微光栅化（把 CAD 网格渲染成裁剪图） | `pip install --no-build-isolation git+.../nvdiffrast.git` |
| **mycpp** | 位姿聚类（旋转网格生成） | `cmake + make`（Boost/Eigen/pybind11） |

`build_ext.sh` 统一处理，设置 `TORCH_CUDA_ARCH_LIST=8.7`（Orin = SM 8.7）与
`CUDA_HOME`，日志写入 `logs/`。

### 8.3 稳健导入链

官方 `Utils.py` 对可选依赖做了 import guard（`kaolin`、`warp`、`kornia`、`mycpp`、
`bundlesdf.mycuda` 均为 try/except）。本项目遵循官方设计：

- **kaolin** 仅 model-free 路径需要，本项目不装即可（`Utils` 顶层不 import kaolin）；
- **warp-lang** 用于深度滤波（`erode_depth` / `bilateral_filter_depth`），
  已列入 `requirements.txt`；缺失时相关函数不可用；
- **triton** 在 Jetson 无 aarch64 wheel，但 FoundationPose **不依赖 triton**，
  无 SAM3 那样的补丁需求。

### 8.4 权重与数据

- **权重**：官方托管在 Google Drive 文件夹（readme「Data prepare」）。
  `download.sh` 优先用直链镜像（`FP_WEIGHTS_BASE`），否则用 `gdown` 拉取；
- **示例数据**：同样来自 Google Drive，`download.sh` 一并处理；
- 两者均可手动下载后按目录结构放入 `weights/` 与 `demo_data/`。

### 8.5 安装踩坑记录

1. **`--no-build-isolation` 必需**：编译 pytorch3d/nvdiffrast 时必须能看到本
   venv 的 torch，否则会另起隔离环境导致 CUDA 版本错配。
2. **`CUDA_HOME` 与 `TORCH_CUDA_ARCH_LIST`**：Jetson 上需显式指定
   `/usr/local/cuda` 与 `8.7`，否则可能编出不含 GPU 代码的库。
3. **Boost/Eigen/pybind11**：mycpp 编译依赖系统级 `libeigen3-dev`、
   `libboost-all-dev`，缺失时 `build_ext.sh mycpp` 会失败。
4. **内存**：编译原生扩展时 `MAX_JOBS` 不宜过大（AGX Orin 上建议 4），
   否则并行编译可能耗尽统一内存。
5. **`python3.10 -m venv` 报 ensurepip 不可用**：JetPack 默认没装
   `python3.10-venv` 包。`setup.sh` 会探测 `ensurepip`，缺失时自动改用
   `--without-pip` + `get-pip.py` 引导 pip，无需 sudo，属正常路径，不是错误。
6. **NVIDIA 的 PyTorch 直链已失效**：旧地址
   `developer.download.nvidia.com/compute/redist/jp/v61/pytorch/*.whl`
   现已整体返回 **HTTP 404**（与网络无关）。`setup.sh` 因此改为两级来源：
   优先用本地 `wheels/` 目录中已下好的 wheel，缺失时从 Jetson 社区镜像
   `pypi.jetson-ai-lab.io/jp6/cu126` 安装。若镜像也不可达，手动把
   `torch-2.8.0-cp310-cp310-linux_aarch64.whl` 与
   `torchvision-0.23.0-cp310-cp310-linux_aarch64.whl` 放进 `wheels/` 再跑
   `setup.sh` 即可（会自动优先用本地 wheel）。
7. **`numpy` 必须锁 `<2`**：NVIDIA 的 Jetson torch 2.8.0 wheel 是针对 NumPy 1.x
   编译的。若解析到 NumPy 2.x，`import torch` 会打印
   `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
   并出现 `Failed to initialize NumPy`，`tensor.numpy()` 等互操作随之失效。
   `requirements.txt` 已写死 `numpy<2`（本机实测 1.26.4）。
8. **长任务不要 `>> log 2>&1`**：pip / gdown 只在 stdout 是 tty 时才显示实时
   进度条，直接重定向到日志会让终端长时间一片空白（216MB 的 torch wheel、
   2.6G 的示例数据、30-60 分钟的 pytorch3d 编译都属此类）。
   `sh/_common.sh` 提供 `run_logged`，用 `script(pty)` 包一层，做到
   **终端实时可见 + 日志照样落盘**；`setup.sh` / `build_ext.sh` / `download.sh`
   的所有长任务都走它。

---

## 9. 性能实测

> 以下为结构说明；具体数值因物体/分辨率/迭代次数而异，请以本机 `--bench` 实测为准。

| 阶段 | 说明 | 可复用性 |
|------|------|---------|
| 导入三方库 | torch/pytorch3d/nvdiffrast 等 | 常驻后省去 |
| 网络加载 | Scorer/Refiner 权重 | **常驻后省去** |
| glctx 初始化 | CUDA 光栅化上下文 | **常驻后省去** |
| 首帧注册 | 多假设渲染 + 精修 + 打分（`est_refine_iter` 次） | 每个物体一次 |
| 逐帧跟踪 | 渲染 + 精修（`track_refine_iter` 次） | 每帧 |

基准命令：

```bash
./sh/run.sh --video demo_data/mustard0 --mesh demo_data/mustard0/mesh/textured_simple.obj \
            --bench --repeat 3
```

**本机实测（AGX Orin 64GB，`demo_data/mustard0`，`est_refine_iter=5` / `track_refine_iter=2`）：**

| 项目 | 数值 |
|------|------|
| 常驻服务启动（含加载权重 + glctx） | 约 7s |
| 首帧注册 | **约 6.3–6.9s**（252 个旋转假设 + 5 次精修 + 打分） |
| 逐帧跟踪 | **约 78–164ms**（均值约 120ms） |
| 显存（分配 / 峰值） | 约 0.32 GiB / 4.32 GiB |

> 常驻服务下单次请求（3 帧）端到端约 17.7s，其中绝大部分是首帧注册；
> 跟踪帧稳定在 ~100ms 量级。

---

## 10. 常见问题

**Q1. `ModuleNotFoundError: pytorch3d` / `nvdiffrast`**
原生扩展未编译：执行 `./sh/build_ext.sh`，或单独
`./sh/build_ext.sh pytorch3d` / `nvdiffrast`。查看 `logs/build_*.log`。

**Q2. `import mycpp` 失败 / 位姿聚类报错**
mycpp 未构建：`./sh/build_ext.sh mycpp`。需先安装
`sudo apt-get install -y libeigen3-dev libboost-all-dev cmake build-essential`。

**Q3. 权重缺失**
`./sh/download.sh`。若 Google Drive 不可达，可设置 `FP_WEIGHTS_BASE`
指向直链镜像，或手动下载放入 `weights/<时间戳>/`。

**Q4. 结果不合理 / 位姿漂移**
- 确认 `depth/` 单位（应为 **毫米** 的 uint16，或米）；
- 确认 `cam_K.txt` 内参正确，相机与深度对齐；
- 首帧掩码要覆盖目标且尽量贴合；
- 参考官方 issue #44。

**Q5. 显存不足（OOM）**
调小 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`；减少
`--est-refine-iter` / `--track-refine-iter`；`./sh/serve.sh stop` 释放常驻服务。

**Q6. `import torch` 报找不到 `.so`**
`LD_LIBRARY_PATH` 没设。用 `./sh/run.sh`，或手动
`export LD_LIBRARY_PATH=$PWD/venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`。

**Q7. 想用 model-free（无 CAD，仅参考图）**
需额外构建 BundleSDF 的 CUDA 算子与（可选）kaolin，本项目默认不启用：
`cd framework_src/bundlesdf/mycuda && pip install --no-build-isolation -e .`。

**Q8. TRT 为什么没有明显加速？**
见 [第 5 节](#5-tensorrt-加速可选)：瓶颈在渲染与访存，TRT 只接管网络前向。

---

## 11. 修复与验证记录

本项目在 Jetson AGX Orin 上完成端到端部署后，对以下问题做了修复与验证：

### 11.1 `trt_runner.py` 文件名损坏（已修复）
早期一次操作中 `py/trt_runner.py` 被误建为**目录**，真实脚本落到了错误的
文件名 `py/trt_runner.py</path`，导致 `fp_trt_infer.py` 的
`from trt_runner import TRTEngine` 无法导入。现已：

- 删除误建的空目录 `py/trt_runner.py<`；
- 将脚本还原为正确的 `py/trt_runner.py`（141 行 `TRTEngine` 封装）。

> TRT 路径仍属**可选**：venv 未装 `tensorrt` 时仅该路径不可用，
> PyTorch 主路径（`sh/run.sh`）完全不受影响。

### 11.2 本次补齐的文件（使 README 描述全部落地）

| 文件 | 作用 | 验证结果 |
|------|------|---------|
| `py/fp_server.py` | 常驻服务（Unix socket + JSON 协议） | `serve.sh start/status/stop` 正常，ping 返回设备与 uptime |
| `py/fp_batch.py` | 批量多场景处理 + CSV 摘要 | 输出可视化 PNG、每场景位姿目录与 CSV |
| `sh/check_env.sh` | shell 层环境检查 | 13 项全部通过（TensorRT 为可选提示） |

### 11.3 常驻服务「省去重复加载」验证
启动服务后执行 `./sh/run.sh`，客户端自动转发到服务：
注册 + 跟踪的**推理结果与本地运行完全一致**，且省去了每次约十数秒的
网络加载与 glctx 初始化开销（服务内网络加载耗时显示为 `0.0s`）。

### 11.4 建议的完整自检流程

```bash
./sh/setup.sh          # 一次性初始化（venv + 依赖 + 扩展 + 权重）
./sh/check_env.sh      # shell 层：工具链 / 扩展 / 包 / 权重
./venv310/bin/python py/selfcheck.py   # Python 层：import / CUDA / 资源
./sh/run.sh --video demo_data/mustard0 \
            --mesh demo_data/mustard0/mesh/textured_simple.obj --max-frames 3
```

---

## 12. 许可

模型与代码来自 NVIDIA，遵循其
[NVIDIA Source Code License](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE)。
本项目仅为 Jetson 部署适配。
