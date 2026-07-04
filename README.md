[English](README.en.md) | 简体中文

# TensorRT-DETR

TensorRT-DETR 是面向 NVIDIA GPU 的 C++/CUDA/TensorRT 推理部署库，提供 C++ 与 Python 两套接口，覆盖目标检测、实例分割、姿态估计任务。


## 依赖

- CUDA
- TensorRT
- Python 绑定可选依赖：Python Development、pybind11、pip

## 编译安装

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
cmake --build build -j$(nproc) --config Release --target install
pip install dist/trtdetr-*.whl
```

启用 `BUILD_PYTHON=ON` 后，构建会生成 `dist/trtdetr-*.whl`

## 模型转换

本项目按导出方式将支持的模型分为两类：

- **可直接用 `trtexec` 转换**：RF-DETR、YOLOv26 / YOLO26 等官方已提供“单输入 ONNX”的模型。
- **需参考 [`assets/export/export_onnx.py`](assets/export/export_onnx.py) 转换**：RT-DETR、D-FINE、DEIM / DEIMv2、EdgeCrafter 等 DETR 系模型，需要去掉官方导出脚本中的 `orig_target_sizes` 输入，改由本项目在推理侧通过 letterbox 的 `Transform` 反变换。

完整的模型对照表与转换步骤见 [assets/export/README.md](assets/export/README.md)。导出 ONNX 后，可继续使用 TensorRT 工具链构建 engine，再由本项目进行推理部署。

## Python 使用

```python
import cv2
from trtdetr import TRTDETR

model = TRTDETR("model.engine", task="detect", profile=True, swap_rb=True, conf_thresh=0.25)
image = cv2.imread("image.jpg")
result = model.predict(image)
print(result)
```

python示例见 [examples/python](examples/python/)。

## C++ 使用

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

## 致谢

本项目主要代码来源于 [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO)，模型转换主要使用 [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)，并在相关推理框架与模型导出流程基础上进行了整理和适配。感谢相关开源项目对 TensorRT 部署生态的贡献。
