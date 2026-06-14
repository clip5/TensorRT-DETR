from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv

from . import c_lib_wrap as C

__all__ = ["TRTDETR"]


class TRTDETR:
    """
    TensorRT-DETR unified interface

    One-stop gateway to TensorRT-accelerated vision models for five tasks:

        * detect   - Object Detection
        * segment  - Instance Segmentation
        * classify - Image Classification
        * pose     - Pose Estimation
        * obb      - Oriented Bounding-Box

    Attributes:
        task_map (Dict[str, Type[C.model.BaseModel]]): Task-to-model mapping.
        _model (C.model.BaseModel): Underlying TensorRT model instance.
        _task (str): Current task name.

    Examples:
        >>> from trtdetr import TRTDETR
        >>> det = TRTDETR("yolo11n.engine", task="detect")
        >>> seg = TRTDETR("yolo11n-seg.engine", task="segment", device=1)
        >>> cls = TRTDETR("yolo11n-cls.engine", task="classify",
        ...               mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    """

    def __init__(
        self,
        model: Union[str, Path],
        task: str,
        device: Optional[int] = 0,
        swap_rb: Optional[bool] = True,
        profile: Optional[bool] = False,
        border_value: Optional[float] = None,
        conf_thresh: Optional[float] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize TRTDETR model

        Args:
            model (str | Path): Path to TensorRT engine (*.engine).
            task (str): Task in {'detect', 'segment', 'classify', 'pose', 'obb'}.
            device (int, optional): GPU device id. Default 0.
            swap_rb (bool, optional): Swap R<->B during preprocessing. Default True.
            profile (bool, optional): Enable latency profiler. Default False.
            border_value (float, optional): Pad value for letterbox. Default None (114).
            conf_thresh (float, optional): Confidence threshold used by post-processing. Default None (0.25).
            mean (Tuple[float, float, float], optional): Normalization mean. Must pair with `std`.
            std (Tuple[float, float, float], optional): Normalization std. Must pair with `mean`.
            input_size (Tuple[int, int], optional): Force **fixed input resolution** (width, height)
                                                   for preprocessing to avoid dynamic-shape overhead.
                                                   Only useful when **real input size is constant**
                                                   (e.g. video analysis). Does **not** change model
                                                   native shape. Default None (dynamic).
        """
        option = C.option.InferOption()
        option.set_device_id(device)
        if swap_rb:
            option.enable_swap_rb()
        if profile:
            option.enable_profile()
        if border_value is not None:
            option.set_border_value(border_value)
        if conf_thresh is not None:
            option.set_conf_thresh(conf_thresh)
        if mean is not None and std is not None:
            option.set_normalize_params(mean, std)
        if input_size is not None:
            option.set_input_dimensions(input_size[0], input_size[1])

        self._task = task
        self._model = self.task_map[task](model, option)

    @property
    def task_map(self) -> Dict[str, Any]:
        """Map head to model classes."""
        return {
            "classify": C.model.ClassifyModel,
            "detect": C.model.DetectModel,
            "segment": C.model.SegmentModel,
            "pose": C.model.PoseModel,
            "obb": C.model.OBBModel,
        }

    @property
    def batch(self) -> int:
        """Get model batch size."""
        return self._model.batch

    def __copy__(self) -> None:
        cls = self.__class__.__name__
        raise NotImplementedError(f"Model '{cls}' does not support copy, use clone() instead.")

    def __deepcopy__(self, memo: dict) -> None:
        cls = self.__class__.__name__
        raise NotImplementedError(f"Model '{cls}' does not support deepcopy, create a new instance instead.")

    def __call__(
        self,
        source: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
    ) -> Union[List[C.result.BaseRes], List[List[C.result.BaseRes]]]:
        """
        Predict on an image or batch of images.

        Args:
            source (str | Path | np.ndarray | List[str | Path | np.ndarray]): Source image or batch of images.

        Returns:
            List[C.result.BaseRes] | List[List[C.result.BaseRes]]: Prediction results for the input image(s).
        """
        return self.predict(source)

    def clone(self) -> "TRTDETR":
        """Return a shallow-copy model instance that shares the same TensorRT engine."""
        new_obj = self.__class__.__new__(self.__class__)
        new_obj._task = self._task
        new_obj._model = self._model.clone()
        return new_obj

    def predict(
        self,
        source: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
    ) -> Union[
        Union[sv.Detections, sv.KeyPoints, sv.Classifications],
        List[Union[sv.Detections, sv.KeyPoints, sv.Classifications]],
    ]:
        """
        Predict on an image or batch of images.

        Args:
            source (str | Path | np.ndarray | List[str | Path | np.ndarray]): Source image or batch of images.

        Returns:
            Union[sv.Detections, sv.KeyPoints, sv.Classifications] | List[Union[sv.Detections, sv.KeyPoints, sv.Classifications]]: Prediction results for the input image(s).
        """
        if isinstance(source, list):
            if all(isinstance(s, (str, Path)) for s in source):
                source = [cv2.imread(str(s)) for s in source]
            elif all(isinstance(s, np.ndarray) for s in source):
                pass
            else:
                raise ValueError("source must be a list of str, Path, or np.ndarray")
            batches = [source[i : i + self._model.batch] for i in range(0, len(source), self._model.batch)]
            results = [r for batch in batches for r in self._model.predict(batch)]
            assert len(results) == len(source)
            return [convert_to_sv(res, img.shape[:2]) for res, img in zip(results, source)]
        else:
            if isinstance(source, (str, Path)):
                img = cv2.imread(str(Path(source)))
            elif isinstance(source, np.ndarray):
                img = source
            return convert_to_sv(self._model.predict(img), img.shape[:2])

    def profile(self) -> Tuple[str, str, str]:
        """
        Get latency statistics.

        Requires `profile=True` during initialization.

        Returns:
            Tuple[str, str, str]: (throughput, cpu_latency, gpu_latency)
                throughput  - 'Throughput: 120.14 qps'
                cpu_latency - 'CPU Latency: min = 8.32 ms, max = 8.35 ms, mean = 8.33 ms, ...'
                gpu_latency - 'GPU Latency: min = 8.12 ms, max = 8.15 ms, mean = 8.13 ms, ...'

        Note:
            Returns three empty strings if profiler was disabled.
        """
        return self._model.profile()


def convert_to_sv(result: C.result.BaseRes, img_shape: Tuple[int, int]) -> Union[sv.Detections, sv.KeyPoints, sv.Classifications]:
    """
    Convert C.result.BaseRes -> supervision format (sv.Detections / KeyPoints / Classifications)

    Args:
        result (C.result.BaseRes): The C++ result object to convert.

    Returns:
        Union[sv.Detections, sv.KeyPoints, sv.Classifications]: The converted Supervision object.
    """

    if isinstance(result, C.result.DetectRes):
        return sv.Detections(
            xyxy=result.xyxy,
            confidence=result.confidence,
            class_id=result.class_id,
        )
    elif isinstance(result, C.result.OBBRes):
        return sv.Detections(
            xyxy=result.xyxy,
            confidence=result.confidence,
            class_id=result.class_id,
            data={
                sv.config.ORIENTED_BOX_COORDINATES: result.xyxyxyxy,
            },
        )
    elif isinstance(result, C.result.SegmentRes):
        return sv.Detections(
            xyxy=result.xyxy,
            confidence=result.confidence,
            class_id=result.class_id,
            mask=paste_masks_in_image(result.masks, result.xyxy, img_shape),
        )
    elif isinstance(result, C.result.PoseRes):
        xy, conf = split_xy_conf(result.kpts)

        return sv.KeyPoints(
            xy=xy,
            confidence=conf,
            class_id=result.class_id,
        )
    elif isinstance(result, C.result.ClassifyRes):
        return sv.Classifications(
            class_id=result.class_id,
            confidence=result.confidence,
        )
    else:
        raise ValueError(f"Unknown result type: {type(result)}")


def split_xy_conf(arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Split the input array into two parts: coordinates and confidence.

    When the last dimension is 2 -> xy=arr,       conf=None
    When the last dimension is 3 -> xy=arr[...,:2], conf=arr[...,2]
    Convert all values to int type.

    Args:
        arr (np.ndarray): Input array where the last dimension is either 2 or 3.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: A tuple containing the coordinate array and the confidence array (may be None).
    """
    if arr.shape[-1] == 3:
        xy, conf = np.split(arr, [2], axis=-1)  # xy:(...,2)  conf:(...,1)
        return xy.astype(int), conf.squeeze(-1).astype(float)
    elif arr.shape[-1] == 2:  # shape[-1] == 2
        return arr.astype(int), None
    else:
        return np.empty((0, 0, 2), dtype=int), None


def paste_masks_in_image(masks: np.ndarray, boxes: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Restore DETR-style low-resolution whole-image masks to the original image shape.

    The C++ segment post-processing returns masks in the low-resolution model input
    space, for example [N, 160, 160] for a 640x640 model input. These masks are not
    bbox-local masks, so they must be de-letterboxed before resizing to the original
    image.
    """
    if masks.size == 0:
        return None

    im_h, im_w = img_shape
    mask_h, mask_w = masks.shape[-2:]
    model_w = mask_w * 4
    model_h = mask_h * 4

    scale = min(model_w / im_w, model_h / im_h)
    valid_w = round(scale * im_w)
    valid_h = round(scale * im_h)
    offset_x = (model_w - valid_w) // 2
    offset_y = (model_h - valid_h) // 2

    crop_x = round(offset_x * mask_w / model_w)
    crop_y = round(offset_y * mask_h / model_h)
    crop_w = round(valid_w * mask_w / model_w)
    crop_h = round(valid_h * mask_h / model_h)

    crop_x = max(0, min(crop_x, mask_w - 1))
    crop_y = max(0, min(crop_y, mask_h - 1))
    crop_w = max(1, min(crop_w, mask_w - crop_x))
    crop_h = max(1, min(crop_h, mask_h - crop_y))

    restored_masks = []
    for mask in masks:
        valid_mask = mask[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        restored = cv2.resize(valid_mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR) >= 0.5
        restored_masks.append(restored)
    return np.array(restored_masks, dtype=bool)
