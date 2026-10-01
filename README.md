[English](README.en.md) | 简体中文

# TensorRT-DETR

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C++-17-00599C.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900.svg)]()
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x%20%7C%2010.x-76b900.svg)]()

TensorRT-DETR 是面向 NVIDIA GPU 的 C++/CUDA/TensorRT 推理部署库，提供 C++ 与 Python 两套接口，当前覆盖目标检测、实例分割、姿态估计任务（OBB 支持规划中）。

<div align="center">
  <img src="assets/detect_result.jpg" width="32%">
  <img src="assets/segment_result.jpg" width="32%">
  <img src="assets/pose_result.jpg" width="32%">
  <p><em>检测 · 分割 · 姿态估计 — 单张 RTX 4070 Ti SUPER 上最高 255 QPS</em></p>
</div>

## ✨ 特性

- 🚀 任务覆盖：检测 / 分割 / 姿态（OBB 支持规划中）
- ⚡ 基于 [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) 推理内核：CUDA Stream + CUDA Graph 加速、GPU 端 letterbox 预处理
- 🧠 多种内存策略：Device / Unified / Mapped 按需切换
- 🔗 C++ 与 Python 双接口，pybind11 wheel 一键安装
- 📦 输出直接对接 [supervision](https://github.com/roboflow/supervision)，可视化零样板代码
- 🎯 针对 DETR 系模型（[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) 导出）适配：ONNX 仅需 `images` 单输入

## 🚀 快速开始（Python）

```bash
pip install dist/trtdetr-*.whl
```

```python
import cv2
from trtdetr import TRTDETR

model = TRTDETR("model.engine", task="detect", profile=True, swap_rb=True, conf_thresh=0.25)
image = cv2.imread("image.jpg")
result = model.predict(image)
print(result)
```

更多 Python 示例见 [examples/python](examples/python/)。

## 📊 性能

在 RTX 4070 Ti SUPER 上实测（batch=1，FP16 engine，输入 640×640）：

| 任务 | 模型 | 吞吐 (QPS) | CPU 延迟 | GPU 延迟 |
|------|------|-----------|---------|---------|
| 检测 | ecdet_s | 239 | 4.18 ms | 4.16 ms |
| 分割 | ecseg_s | 115 | 8.73 ms | 8.72 ms |
| 姿态 | ecpose_s | 255 | 3.92 ms | 3.91 ms |

## 📑 目录

- [✨ 特性](#-特性)
- [🚀 快速开始](#-快速开始python)
- [📊 性能](#-性能)
- [依赖](#依赖)
- [🔨 编译安装](#-编译安装)
- [📦 模型转换](#-模型转换)
- [🧩 C++ 示例](#-c-示例)
- [🐍 Python 使用](#-python-使用)
- [🔧 C++ 使用](#-c-使用)
- [许可证](#许可证)
- [🙏 致谢](#-致谢)

## 依赖

- CUDA
- TensorRT
- CMake >= 3.18
- C++17 编译器
- Python 绑定可选依赖：Python Development、pybind11、pip

## 🔨 编译安装

仅编译 C++ 库：

```bash
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build -j$(nproc) --config Release --target install
```

编译 Python 绑定并生成 wheel：

```bash
pip install "pybind11[global]"
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DBUILD_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build -j$(nproc) --config Release
pip install dist/trtdetr-*.whl
```

启用 `BUILD_PYTHON=ON` 后，构建会生成 `dist/trtdetr-*.whl`

## 📦 模型转换

本项目支持的模型主要基于 [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) 转换得到。转换模型时，对 EdgeCrafter 的 Python 导出流程做了部分修改，参考 [assets/export_onnx.py](assets/export_onnx.py) 脚本：导出的 ONNX 仅保留图像输入 `images`，不再额外输入原图尺寸等信息。

导出 ONNX 后，可继续使用 TensorRT 工具链构建 engine，再由本项目进行推理部署。

## 🧩 C++ 示例

从仓库根目录统一编译示例：

```bash
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc) --config Release --target detect segment pose mutli_thread
```

也可以按需关闭部分示例：

```bash
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_EXAMPLE_DETECT=ON \
  -DBUILD_EXAMPLE_SEGMENT=OFF \
  -DBUILD_EXAMPLE_POSE=OFF \
  -DBUILD_EXAMPLE_MULTI_THREAD=OFF
```

更多说明见 [examples/README.md](examples/README.md)。

## 🐍 Python 使用

```python
import cv2
from trtdetr import TRTDETR

model = TRTDETR("model.engine", task="detect", profile=True, swap_rb=True, conf_thresh=0.25)
image = cv2.imread("image.jpg")
result = model.predict(image)
print(result)
```

Python 示例见 [examples/python](examples/python/)。

## 🔧 C++ 使用

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "trtdetr.hpp"

int main() {
    trtdetr::InferOption option;
    option.enableSwapRB();
    option.setConfThresh(0.5f);

    trtdetr::DetectModel model("model.engine", option);
    cv::Mat image = cv::imread("image.jpg");
    trtdetr::Image input(image.data, image.cols, image.rows);

    auto result = model.predict(input);
    std::cout << result << std::endl;
    return 0;
}
```

cpp示例见 [examples/cpp](examples/cpp/)。

## 许可证

本项目使用 GPL-3.0 许可证，详见 [LICENSE](LICENSE)。

## 🙏 致谢

本项目主要代码来源于 [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO)，模型转换主要使用 [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)，并在相关推理框架与模型导出流程基础上进行了整理和适配。感谢相关开源项目对 TensorRT 部署生态的贡献。
