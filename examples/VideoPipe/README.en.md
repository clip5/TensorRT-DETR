English | [简体中文](README.md)

# Video Analysis Example

This example uses the YOLO11n model to demonstrate how to integrate the TensorRT-DETR Deploy module into [VideoPipe](https://github.com/sherlockchou86/VideoPipe) for video analysis.

[yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，[demo0.mp4](https://www.ilanzou.com/s/yhUyq8f3)，[demo1.mp4](https://www.ilanzou.com/s/aIhyq8ET)

Please download the required `yolo11n.pt` model file and test video through the provided link, and save both to the `workspace` folder.

## Model Export

> [!IMPORTANT]
>
> Use the `trtdetr-export` tool package that comes with the project to export the ONNX model suitable for inference in this project and build it into a TensorRT engine.


## Project Execution

1. Make sure that the project has been compiled according to the project documentation and the [`VideoPipe` compilation and debugging](https://github.com/sherlockchou86/VideoPipe/blob/master/README.md#compilation-and-debugging) (only the default five steps need to be executed, without adding any other compilation options).

2. Compile the project into an executable:

    ```bash
    cmake -S . -B build -D VIDEOPIPE_PATH="/path/to/your/VideoPipe"
    cmake --build . -j8 --config Release
    ```

    After compilation, the executable file will be generated in the `workspace` folder of the project root directory.

3. Run the following command for inference:

    ```bash
    cd workspace
    ./PipeDemo
    ```

<div align="center">
    <p>
        <img width="100%" src="../../assets/videopipe.png">
    </p>
</div>
