English | [简体中文](README.md)

# TensorRT-DETR

TensorRT-DETR is a C++/CUDA/TensorRT inference deployment library for NVIDIA GPUs. It provides C++ and Python APIs for detection, instance segmentation, pose estimation.


## Requirements

- CUDA
- TensorRT
- CMake >= 3.18
- C++17 compiler
- Optional Python binding dependencies: Python Development, pybind11, pip

## Build and Install

Build the C++ library only:

```bash
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build -j$(nproc) --config Release --target install
```

Build Python bindings and generate a wheel:

```bash
pip install "pybind11[global]"
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DBUILD_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build -j$(nproc) --config Release --target install
pip install dist/trtdetr-*.whl
```

With `BUILD_PYTHON=ON`, the build generates `dist/trtdetr-*.whl`

## Model Conversion

The models supported by this project are mainly converted from [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter). The Python export flow in EdgeCrafter has been partially modified for model conversion; see [assets/export_onnx.py](assets/export_onnx.py). The exported ONNX keeps only the image input `images` and does not require extra inputs such as the original image size.

After exporting ONNX, you can continue using the TensorRT toolchain to build an engine and deploy inference with this project.

## Python Usage

```python
import cv2
from trtdetr import TRTDETR

model = TRTDETR("model.engine", task="detect", profile=True, swap_rb=True, conf_thresh=0.25)
image = cv2.imread("image.jpg")
result = model.predict(image)
print(result)
```

python examples are available under [examples/python](examples/python/).

## C++ Usage

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
cpp examples are available under [examples/cpp](examples/cpp/).

## License

This project is licensed under GPL-3.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

This project is mainly derived from [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO), and the model conversion flow mainly references [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter). It has been reorganized and adapted based on the related inference framework and model export flow. Thanks to the related open-source projects for their contributions to the TensorRT deployment ecosystem.
