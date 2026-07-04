# 模型转换说明

本目录提供 TensorRT-DETR 常用模型的 ONNX 导出与 TensorRT engine 构建流程。不同模型系列的导出方式略有差异，请先根据下表定位到自己使用的模型，再按对应章节操作。

## 转换方式速查表

| 模型系列 | 官方仓库 | 导出方式 | 是否需要 [`export_onnx.py`](export_onnx.py) |
| --- | --- | --- | --- |
| RF-DETR | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | 直接使用官方脚本导出 ONNX，`trtexec` 构建 engine | 否，见[第一节](#一直接使用-trtexec-转换的模型) |
| YOLOv26 / YOLO26 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 直接使用官方脚本导出 ONNX，`trtexec` 构建 engine | 否，见[第一节](#一直接使用-trtexec-转换的模型) |
| RT-DETR | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | 需去除 `orig_target_sizes` 输入后再导出 | 是，见[第二节](#二需要参考-export_onnxpy-转换的模型) |
| D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | 同上 | 是，见[第二节](#二需要参考-export_onnxpy-转换的模型) |
| DEIM / DEIMv2 | [ShihuaHuang95/DEIM](https://github.com/ShihuaHuang95/DEIM) | 同上 | 是，见[第二节](#二需要参考-export_onnxpy-转换的模型) |
| EdgeCrafter | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | 同上 | 是，见[第二节](#二需要参考-export_onnxpy-转换的模型) |

> 判定原则：**ONNX 只有 `images` 一个输入且输出为 `labels/boxes/scores`** 的模型属于第一类，可直接 `trtexec`；若官方导出脚本会额外把 `orig_target_sizes` 送进 postprocessor，则属于第二类，需要用本目录的 `export_onnx.py` 去掉该输入。

## 一、直接使用 trtexec 转换的模型

**适用模型**：RF-DETR、YOLOv26 / YOLO26 等官方仓库已提供“单输入 ONNX”（输入只有图像，输出为 `labels/boxes/scores`，无需额外后处理输入）的模型。

这类模型无需本目录下的 `export_onnx.py`，导出 ONNX 后使用 TensorRT 自带的 `trtexec` 工具直接构建 engine 即可。

示例（以 FP16 为例）：

```bash
# ONNX -> TensorRT engine（静态 batch）
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.engine \
  --fp16 \
  --memPoolSize=workspace:4096

# 若需要动态 batch，请在导出 ONNX 时开启动态维度，然后：
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640
```

> RF-DETR / YOLO26 的 ONNX 导出方式请参考各自官方仓库的文档；本项目仅负责部署侧的 engine 构建与推理。

## 二、需要参考 `export_onnx.py` 转换的模型

**适用模型**：RT-DETR、D-FINE、DEIM / DEIMv2、EdgeCrafter 等基于 DETR 结构的模型（含但不限于以上四类）。

这些模型的官方导出脚本默认会把 **原图尺寸 `orig_target_sizes`** 作为第二个输入送入 postprocessor。这会导致 ONNX 有两个输入，与本项目 C++/Python 推理管线（只喂入 `images`）不兼容。

因此本目录提供了 [`export_onnx.py`](export_onnx.py)：把 postprocessor 中的 `orig_target_sizes` 固定为 `[1.0, 1.0]`（相当于不做原图坐标反变换），由本项目在推理时通过 letterbox 的 `Transform` 自行映射回原图坐标。

使用步骤：

1. 将 `export_onnx.py` 放到对应模型仓库根目录下（或按 `sys.path.insert` 的相对路径调整位置），保证脚本可以 `from engine.core import YAMLConfig`。
2. 使用官方权重进行导出：

   ```bash
   python export_onnx.py \
     -c configs/dfine/dfine_hgnetv2_l_coco.yml \
     -r path/to/checkpoint.pth \
     --check --simplify
   ```

   参数说明：
   - `-c/--config`：模型对应的 YAML 配置。
   - `-r/--resume`：训练得到的 `.pth` 权重；导出结果会保存为同名 `.onnx`。
   - `--opset`：ONNX opset，默认 18。
   - `--check`：使用 `onnx.checker` 校验模型。
   - `--simplify`：使用 `onnxsim` 简化模型（推荐开启）。

3. 得到 ONNX 后，同样使用 `trtexec` 构建 engine：

   ```bash
   trtexec \
     --onnx=model.onnx \
     --saveEngine=model.engine \
     --fp16 \
     --memPoolSize=workspace:4096
   ```
