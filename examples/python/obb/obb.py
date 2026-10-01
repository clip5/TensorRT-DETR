"""
Copyright (c) 2025. All rights reserved.
=============================================================================================
File    :   obb.py
Version :   6.4.0
Date    :   2025/10/01 21:40:00
Desc    :   OBB 示例（旋转目标检测）
=============================================================================================
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from trtdetr import TRTDETR


def main():
    parser = argparse.ArgumentParser(description='TensorRT-DETR inference example.')
    parser.add_argument('-e', '--engine', required=True, type=str, help='The serialized TensorRT engine.')
    parser.add_argument('-i', '--input', required=True, type=str, help="Path to the image or directory to process.")
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory where to save the visualization results.')
    parser.add_argument(
        "-l", "--labels", default="./labels.txt", help="File to use for reading the class labels from, default: ./labels.txt"
    )

    args = parser.parse_args()

    if args.output and not args.labels:
        raise ValueError("Please provide a labels file using -l or --labels.")

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        label_annotator = sv.LabelAnnotator()
        class_name = [line.strip() for line in open(args.labels, "r")]

    input_path = Path(args.input)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]

    model = TRTDETR(args.engine, task="obb", swap_rb=True, profile=True, conf_thresh=0.5)

    def annotate(image, result):
        # result.data[sv.config.ORIENTED_BOX_COORDINATES] 是 (N, 4, 2) 的旋转框四角点
        polygons = np.asarray(result.data[sv.config.ORIENTED_BOX_COORDINATES]).reshape(-1, 4, 2)
        annotated = image.copy()
        for poly in polygons:
            pts = poly.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (251, 81, 163), 2, cv2.LINE_AA)
        labels = [f"{class_name[int(cls)]} {conf:.3f}" for cls, conf in zip(result.class_id, result.confidence)]
        return label_annotator.annotate(scene=annotated, detections=result, labels=labels)

    if input_path.is_dir():
        images, image_names = [], []
        for ext in extensions:
            images.extend([cv2.imread(str(image_path)) for image_path in input_path.glob(ext)])
            image_names.extend([image_path.name for image_path in input_path.glob(ext)])
        if not images:
            raise ValueError(f"No images found in directory: {input_path}")
        results = model.predict(images)
        if args.output:
            for image_name, image, result in zip(image_names, images, results):
                annotated_frame = annotate(image, result)
                output_file = output_dir / image_name
                cv2.imwrite(str(output_file), annotated_frame)
    elif input_path.is_file():
        file_ext = f"*.{input_path.suffix.lower()[1:]}"
        if file_ext not in extensions:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        image = cv2.imread(str(input_path))
        result = model.predict(image)
        if args.output:
            annotated_frame = annotate(image, result)
            output_file = output_dir / input_path.name
            cv2.imwrite(str(output_file), annotated_frame)

    throughput, cpu_latency, gpu_latency = model.profile()
    print(throughput)
    print(cpu_latency)
    print(gpu_latency)


if __name__ == '__main__':
    main()
