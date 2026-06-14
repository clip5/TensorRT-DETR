# Examples

本目录包含 TensorRT-DETR 的示例资源和测试入口。项目主要推理框架来源于 TensorRT-YOLO，并针对 DETR 风格输出做了适配。

## 目录结构

```text
examples/
├── cpp/                 # C++ 示例源码
│   ├── detect/
│   ├── segment/
│   ├── pose/
│   └── mutli_thread/
├── python/              # Python 示例源码
│   ├── detect/
│   ├── segment/
│   ├── pose/
│   └── mutli_thread/
├── nndeploy/            # nndeploy 工作流示例
└── VideoPipe/           # VideoPipe 集成示例
```

任务目录保留模型、图片、标签和输出目录；源码统一放在 `examples/cpp` 和 `examples/python`。

## C++ 示例编译

推荐从仓库根目录统一编译：

```bash
cmake -S . -B build -DTRT_PATH=/path/to/tensorrt -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc) --config Release --target detect segment pose mutli_thread
```

也可以只开启部分示例：

```bash
cmake -S . -B build \
  -DTRT_PATH=/path/to/tensorrt \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_EXAMPLE_DETECT=ON \
  -DBUILD_EXAMPLE_SEGMENT=OFF \
  -DBUILD_EXAMPLE_POSE=OFF \
  -DBUILD_EXAMPLE_MULTI_THREAD=OFF
```


## Python 示例

需要先构建并安装 Python 包：

```bash
cmake -S . -B build -DTRT_PATH=/path/to/tensorrt -DBUILD_PYTHON=ON
cmake --build build -j$(nproc) --config Release
pip install dist/trtdetr-*.whl
```

然后在对应任务目录执行 Python 示例，例如：

```bash
cd examples/detect
python ../python/detect/detect.py -e models/model.engine -i images -o output -l labels.txt
```
