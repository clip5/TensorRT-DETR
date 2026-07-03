/**
 * @file trtdetr.cpp
 * @brief TensorRT-DETR 模型推理相关类和结构体的实现
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector_functions.hpp>

#include "backend.hpp"
#include "utils/common.hpp"

namespace trtdetr {

Image::Image(void* data, int width, int height) : ptr(data), width(width), height(height), channels(3), pitch(width * sizeof(uint8_t) * 3) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
}

Image::Image(void* data, int width, int height, size_t pitch)
    : ptr(data), width(width), height(height), channels(3), pitch(pitch) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width, height and channels must be positive"));
    }
    if (pitch < static_cast<size_t>(width * sizeof(uint8_t) * channels)) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: pitch must >= width * channels"));
    }
}

Image::Image(void* data, int width, int height, int channels, size_t pitch)
    : ptr(data), width(width), height(height), channels(channels), pitch(pitch) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width, height and channels must be positive"));
    }
    if (pitch < static_cast<size_t>(width * sizeof(uint8_t) * channels)) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: pitch must >= width * channels"));
    }
}

Mask::Mask(int width, int height) : width(width), height(height) {
    if (width < 0 || height < 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Mask: width and height must be positive"));
    }
    data.resize(width * height);
}

std::array<int, 4> Box::xyxy() const {
    return {static_cast<int>(left), static_cast<int>(top), static_cast<int>(right), static_cast<int>(bottom)};
}

std::array<int, 8> RotatedBox::xyxyxyxy() const {
    float cx   = (left + right) * 0.5f;
    float cy   = (top + bottom) * 0.5f;
    float dx   = (right - left) * 0.5f;
    float dy   = (bottom - top) * 0.5f;
    float cosa = std::cos(theta);
    float sina = std::sin(theta);

    auto rot = [&](float px, float py) -> std::pair<int, int> {
        int x = static_cast<int>(std::round(px * cosa - py * sina + cx));
        int y = static_cast<int>(std::round(px * sina + py * cosa + cy));
        return {x, y};
    };

    auto [x1, y1] = rot(-dx, -dy);  // 左上
    auto [x2, y2] = rot(dx, -dy);   // 右上
    auto [x3, y3] = rot(dx, dy);    // 右下
    auto [x4, y4] = rot(-dx, dy);   // 左下

    return {x1, y1, x2, y2, x3, y3, x4, y4};
}

class InferOption::Impl {
public:
    InferConfig getInferConfig() const { return infer_config; }
    void        setDeviceId(int id) { infer_config.device_id = id; }
    void        enableCudaMem() { infer_config.cuda_mem = true; }
    void        enableManagedMemory() { infer_config.enable_managed_memory = true; }
    void        enablePerformanceReport() { infer_config.enable_performance_report = true; }
    void        setInputDimensions(int width, int height) { infer_config.input_shape = make_int2(height, width); }
    void        setDetectVariant(DetectVariant variant) { infer_config.detect_variant = variant; }
    void        enableSwapRB() { infer_config.config.swap_rb = true; }
    void        setBorderValue(float value) { infer_config.config.border_value = value; }
    void        setConfThresh(float thresh) { infer_config.config.conf_thresh = thresh; };
    void        setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) {
        assert(mean.size() == 3 && std.size() == 3 && "ProcessConfig: requires the size of mean and std to be 3.");

        infer_config.config.alpha.x = 1.0 / 255.0f / std[0];
        infer_config.config.alpha.y = 1.0 / 255.0f / std[1];
        infer_config.config.alpha.z = 1.0 / 255.0f / std[2];
        infer_config.config.beta.x  = -mean[0] / std[0];
        infer_config.config.beta.y  = -mean[1] / std[1];
        infer_config.config.beta.z  = -mean[2] / std[2];
    }

private:
    InferConfig infer_config;  // < 推理选项配置
};

InferOption::InferOption() : impl_(std::make_unique<InferOption::Impl>()) {}
InferOption::~InferOption() = default;

void InferOption::setDeviceId(int id) { impl_->setDeviceId(id); }
void InferOption::enableCudaMem() { impl_->enableCudaMem(); }
void InferOption::enableManagedMemory() { impl_->enableManagedMemory(); }
void InferOption::enablePerformanceReport() { impl_->enablePerformanceReport(); }
void InferOption::enableSwapRB() { impl_->enableSwapRB(); }
void InferOption::setBorderValue(float border_value) { impl_->setBorderValue(border_value); }
void InferOption::setConfThresh(float thresh) { impl_->setConfThresh(thresh); }
void InferOption::setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) { impl_->setNormalizeParams(mean, std); }
void InferOption::setInputDimensions(int width, int height) { impl_->setInputDimensions(height, width); }
void InferOption::setDetectVariant(DetectVariant variant) { impl_->setDetectVariant(variant); }

class BaseModel::Impl {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    Impl()  = default;
    ~Impl() = default;

    Impl(const std::string& trt_engine_file, const InferOption& infer_option)
        : backend_(std::make_unique<TrtBackend>(trt_engine_file, infer_option.impl_->getInferConfig())) {
        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_ = std::make_unique<GpuTimer>(backend_->stream);
            infer_cpu_trace_ = std::make_unique<CpuTimer>();
        }
    }

    std::unique_ptr<Impl> clone() const {
        auto clone_impl              = std::make_unique<Impl>();
        clone_impl->backend_         = backend_->clone();
        clone_impl->infer_gpu_trace_ = std::make_unique<GpuTimer>(clone_impl->backend_->stream);
        clone_impl->infer_cpu_trace_ = std::make_unique<CpuTimer>();
        clone_impl->detect_variant_resolved_ = detect_variant_resolved_;  // 继承已解析的检测变体
        return clone_impl;
    }

    std::tuple<std::string, std::string, std::string> performanceReport() {
        if (backend_->infer_config.enable_performance_report) {
            float             throughput = total_request_ / infer_cpu_trace_->totalMilliseconds() * 1000;
            std::stringstream ss;

            // 构建吞吐量字符串
            ss << "Throughput: " << throughput << " qps";
            std::string throughputStr = ss.str();
            ss.str("");  // 清空 stringstream

            auto percentiles = std::vector<float>{90, 95, 99};

            auto getLatencyStr = [&](const auto& trace, const std::string& device) {
                auto result = getPerformanceResult(trace->milliseconds(), {0.90, 0.95, 0.99});
                ss << device << " Latency: min = " << result.min << " ms, max = " << result.max
                   << " ms, mean = " << result.mean << " ms, median = " << result.median << " ms";
                for (int32_t i = 0, n = percentiles.size(); i < n; ++i) {
                    ss << ", percentile(" << percentiles[i] << "%) = " << result.percentiles[i] << " ms";
                }
                std::string output = ss.str();
                ss.str("");  // 清空 stringstream
                return output;
            };

            std::string cpuLatencyStr = getLatencyStr(infer_cpu_trace_, "CPU");
            std::string gpuLatencyStr = getLatencyStr(infer_gpu_trace_, "GPU");

            total_request_ = 0;
            infer_cpu_trace_->reset();
            infer_gpu_trace_->reset();

            return std::make_tuple(throughputStr, cpuLatencyStr, gpuLatencyStr);
        }
        // 性能报告未启用时返回空字符串
        return std::make_tuple("", "", "");
    }

    size_t batch() const {
        return backend_->max_shape.x;
    }

    // 装饰器函数
    template <typename Func, typename ReturnType>
    ReturnType withPerformanceReport(const std::vector<Image>& images, Func func) {
        if (backend_->infer_config.enable_performance_report) {
            total_request_ += (backend_->dynamic ? images.size() : backend_->max_shape.x);
            infer_cpu_trace_->start();
            infer_gpu_trace_->start();
        }

        backend_->infer(images);  // 调用推理方法
        ReturnType result = func(images.size());

        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_->stop();
            infer_cpu_trace_->stop();
        }

        return result;
    }

    // ClassifyModel 的后处理方法实现
    ClassifyRes postProcessClassify(int idx) {
        auto&  tensor_info = backend_->tensor_infos[1];
        float* topk        = static_cast<float*>(tensor_info.buffer->host()) + idx * tensor_info.shape.d[1] * tensor_info.shape.d[2];

        ClassifyRes result;
        result.num = tensor_info.shape.d[1];
        result.scores.reserve(result.num);
        result.classes.reserve(result.num);

        for (int i = 0; i < result.num; ++i) {
            result.scores.push_back(topk[i * tensor_info.shape.d[2]]);
            result.classes.push_back(topk[i * tensor_info.shape.d[2] + 1]);
        }

        return result;
    }

    // DetectModel 的后处理方法实现
    DetectRes postProcessDetect(int idx) {
        // 首次调用时解析变体（缓存到成员）
        if (detect_variant_resolved_ == DetectVariant::Auto) {
            detect_variant_resolved_ = resolveDetectVariant();
        }

        switch (detect_variant_resolved_) {
            case DetectVariant::RFDETR:      return postProcessDetectRFDETR(idx);
            case DetectVariant::YoloEnd2End: return postProcessDetectYoloE2E(idx);
            case DetectVariant::EdgeDETR:
            default:                         return postProcessDetectEdgeDETR(idx);
        }
    }

    // EdgeCrafter / D-FINE / RT-DETR 风格：3 个输出（labels / boxes / scores，boxes 为归一化 xyxy）
    DetectRes postProcessDetectEdgeDETR(int idx) {
        auto& input_tensor = backend_->tensor_infos[0];
        auto& class_tensor = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];

        int    max_num = class_tensor.shape.d[1];
        int64_t*   classes = static_cast<int64_t*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];

        DetectRes result;
        result.num   = 0;
        int box_size = box_tensor.shape.d[2];
        int height = input_tensor.shape.d[2];
        int width  = input_tensor.shape.d[3];
        float conf_thresh = backend_->infer_config.config.conf_thresh;

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(max_num);
        result.scores.reserve(max_num);
        result.classes.reserve(max_num);

        for (int i = 0; i < max_num; ++i) {
            if (scores[i] <= conf_thresh) continue;

            int   base_index = i * box_size;
            float left = boxes[base_index] * width, top = boxes[base_index + 1] * height;
            float right = boxes[base_index + 2] * width, bottom = boxes[base_index + 3] * height;

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);
            ++result.num;
        }

        return result;
    }

    // RF-DETR 风格：2 个输出（pred_boxes 归一化 cxcywh + logits 原始 logit，需要 sigmoid + argmax）
    DetectRes postProcessDetectRFDETR(int idx) {
        auto& input_tensor = backend_->tensor_infos[0];
        // 在 resolveDetectVariant() 中确保了 tensor_infos[1]=pred_boxes, tensor_infos[2]=logits
        auto& box_tensor    = backend_->tensor_infos[1];
        auto& logit_tensor  = backend_->tensor_infos[2];

        int    max_num     = box_tensor.shape.d[1];
        int    num_classes = logit_tensor.shape.d[2];
        float* boxes       = static_cast<float*>(box_tensor.buffer->host()) + idx * max_num * box_tensor.shape.d[2];
        float* logits      = static_cast<float*>(logit_tensor.buffer->host()) + idx * max_num * num_classes;

        DetectRes result;
        result.num        = 0;
        int   height      = input_tensor.shape.d[2];
        int   width       = input_tensor.shape.d[3];
        float conf_thresh = backend_->infer_config.config.conf_thresh;

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(max_num);
        result.scores.reserve(max_num);
        result.classes.reserve(max_num);

        // RF-DETR 的 logits 通道对应训练时的完整类别空间：
        //   - 对于 COCO 预训练权重，num_classes==91 表示原始 COCO category id (0..90，带 gap)，
        //     此时 index 32 = "tie"（领带），并非 80-class 连续序号中的 "sports ball"。
        //     若使用 80-class 的 labels.txt，需要在上层做 91→80 的映射，或改用 91-class 的 labels.txt。
        //   - 自定义训练时，num_classes 等于训练类别数，类别索引直接与用户的 labels.txt 对应。
        // 我们不在这里做任何类别偏移，避免二次错位。
        for (int i = 0; i < max_num; ++i) {
            float* logit_row = logits + i * num_classes;
            int    best_c    = 0;
            float  best_l    = logit_row[0];
            for (int c = 1; c < num_classes; ++c) {
                if (logit_row[c] > best_l) {
                    best_l = logit_row[c];
                    best_c = c;
                }
            }
            float score = 1.0f / (1.0f + std::exp(-best_l));
            if (score <= conf_thresh) continue;

            float cx = boxes[i * 4 + 0] * width;
            float cy = boxes[i * 4 + 1] * height;
            float w  = boxes[i * 4 + 2] * width;
            float h  = boxes[i * 4 + 3] * height;
            float left = cx - w * 0.5f, top = cy - h * 0.5f;
            float right = cx + w * 0.5f, bottom = cy + h * 0.5f;

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(score);
            result.classes.push_back(best_c);
            ++result.num;
        }

        return result;
    }

    // YOLO end-to-end 风格：1 个输出 [B, N, 6]（xyxy 已在模型输入空间像素、score、class）
    DetectRes postProcessDetectYoloE2E(int idx) {
        auto& out_tensor = backend_->tensor_infos[1];

        int    max_num = out_tensor.shape.d[1];
        int    stride  = out_tensor.shape.d[2];  // 通常为 6
        float* data    = static_cast<float*>(out_tensor.buffer->host()) + idx * max_num * stride;

        DetectRes result;
        result.num        = 0;
        float conf_thresh = backend_->infer_config.config.conf_thresh;

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(max_num);
        result.scores.reserve(max_num);
        result.classes.reserve(max_num);

        for (int i = 0; i < max_num; ++i) {
            float* row   = data + i * stride;
            float  score = row[4];
            if (score <= conf_thresh) continue;

            float left   = row[0];  // 已经是模型输入空间像素坐标，不再乘 (W, H)
            float top    = row[1];
            float right  = row[2];
            float bottom = row[3];
            int   cls    = static_cast<int>(row[5]);

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(score);
            result.classes.push_back(cls);
            ++result.num;
        }

        return result;
    }

    // 根据输出张量数量与形状嗅探检测输出变体（不依赖 tensor 名称，兼容不同导出脚本）
    DetectVariant resolveDetectVariant() {
        // 用户显式指定则直接返回
        DetectVariant requested = backend_->infer_config.detect_variant;
        if (requested != DetectVariant::Auto) return requested;

        // 统计输出张量数量（tensor_infos[0] 恒为 input）
        int num_outputs = 0;
        for (auto& ti : backend_->tensor_infos) {
            if (!ti.input) ++num_outputs;
        }

        auto shape_str = [&]() {
            std::stringstream ss;
            for (auto& ti : backend_->tensor_infos) {
                if (ti.input) continue;
                ss << ti.name << "[";
                for (int d = 0; d < ti.shape.nbDims; ++d) {
                    if (d) ss << ",";
                    ss << ti.shape.d[d];
                }
                ss << "] ";
            }
            return ss.str();
        };

        // 1 个输出 & 最后一维为 6  ->  YOLO end-to-end [B, N, 6]
        if (num_outputs == 1) {
            auto& ti = backend_->tensor_infos[1];
            if (ti.shape.nbDims >= 3 && ti.shape.d[ti.shape.nbDims - 1] == 6) {
                return DetectVariant::YoloEnd2End;
            }
            throw std::runtime_error(MAKE_ERROR_MESSAGE(
                "DetectModel: single-output engine but shape is not [B,N,6]; please setDetectVariant explicitly. outputs=" + shape_str()));
        }

        // 2 个输出：按 shape 判定 RF-DETR，一路 [B,N,4](boxes)，一路 [B,N,C>4](logits)，且 N 一致
        if (num_outputs == 2) {
            auto& t1 = backend_->tensor_infos[1];
            auto& t2 = backend_->tensor_infos[2];

            auto looks_like_box    = [](const nvinfer1::Dims& d) { return d.nbDims == 3 && d.d[2] == 4; };
            auto looks_like_logits = [](const nvinfer1::Dims& d) { return d.nbDims == 3 && d.d[2] > 4;  };

            const bool same_n = (t1.shape.nbDims == 3 && t2.shape.nbDims == 3 && t1.shape.d[1] == t2.shape.d[1]);
            if (same_n && looks_like_box(t1.shape) && looks_like_logits(t2.shape)) {
                return DetectVariant::RFDETR;
            }
            if (same_n && looks_like_box(t2.shape) && looks_like_logits(t1.shape)) {
                // 规范化顺序：tensor_infos[1]=boxes, tensor_infos[2]=logits
                std::swap(backend_->tensor_infos[1], backend_->tensor_infos[2]);
                return DetectVariant::RFDETR;
            }
            throw std::runtime_error(MAKE_ERROR_MESSAGE(
                "DetectModel: two-output engine but shapes do not match RF-DETR (expect [B,N,4] + [B,N,C>4]); outputs=" + shape_str()));
        }

        // 3 个输出  ->  EdgeCrafter / D-FINE / RT-DETR
        if (num_outputs == 3) {
            return DetectVariant::EdgeDETR;
        }

        throw std::runtime_error(MAKE_ERROR_MESSAGE(
            "DetectModel: unsupported number of output tensors=" + std::to_string(num_outputs) + "; outputs=" + shape_str()));
    }

    // OBBModel 的后处理方法实现
    OBBRes postProcessOBB(int idx) {
        auto& num_tensor   = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& class_tensor = backend_->tensor_infos[4];

        int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
        float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

        OBBRes result;
        result.num   = num;
        int box_size = box_tensor.shape.d[2];

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(num);
        result.scores.reserve(num);
        result.classes.reserve(num);

        for (int i = 0; i < num; ++i) {
            int   base_index = i * box_size;
            float left = boxes[base_index], top = boxes[base_index + 1];
            float right = boxes[base_index + 2], bottom = boxes[base_index + 3];
            float theta = boxes[base_index + 4];

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
            result.scores.push_back(scores[i]);
            result.classes.push_back(classes[i]);
        }

        return result;
    }

    // SegmentModel 的后处理方法实现
    SegmentRes postProcessSegment(int idx) {
        auto& input_tensor = backend_->tensor_infos[0];
        auto& class_tensor = backend_->tensor_infos[1];
        auto& box_tensor   = backend_->tensor_infos[2];
        auto& score_tensor = backend_->tensor_infos[3];
        auto& mask_tensor  = backend_->tensor_infos[4];
        int   mask_height  = mask_tensor.shape.d[2];
        int   mask_width   = mask_tensor.shape.d[3];

        int max_num = class_tensor.shape.d[1];
        auto label_at = [&](int i) -> int {
            switch (class_tensor.dtype()) {
                case nvinfer1::DataType::kINT32:
                    return static_cast<int*>(class_tensor.buffer->host())[idx * class_tensor.shape.d[1] + i];
                case nvinfer1::DataType::kINT64:
                    return static_cast<int>(static_cast<int64_t*>(class_tensor.buffer->host())[idx * class_tensor.shape.d[1] + i]);
                case nvinfer1::DataType::kFLOAT:
                    return static_cast<int>(static_cast<float*>(class_tensor.buffer->host())[idx * class_tensor.shape.d[1] + i]);
                default:
                    throw std::runtime_error(MAKE_ERROR_MESSAGE("SegmentModel: unsupported labels dtype"));
            }
        };
        float* boxes  = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
        float* scores = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        float* masks  = static_cast<float*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

        SegmentRes result;
        result.num   = 0;
        int box_size = box_tensor.shape.d[2];
        int height   = input_tensor.shape.d[2];
        int width    = input_tensor.shape.d[3];
        float conf_thresh = backend_->infer_config.config.conf_thresh;

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(max_num);
        result.scores.reserve(max_num);
        result.classes.reserve(max_num);
        result.masks.reserve(max_num);

        for (int i = 0; i < max_num; ++i) {
            if (scores[i] <= conf_thresh) continue;

            int   base_index = i * box_size;
            float left = boxes[base_index] * width, top = boxes[base_index + 1] * height;
            float right = boxes[base_index + 2] * width, bottom = boxes[base_index + 3] * height;

            transform.apply(left, top, &left, &top);
            transform.apply(right, bottom, &right, &bottom);

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(label_at(i));

            Mask mask(mask_width, mask_height);
            // Directly copy all mask data without edge cropping
            int start_idx = i * mask_height * mask_width;
            std::memcpy(mask.data.data(), masks + start_idx, mask_height * mask_width * sizeof(float));

            result.masks.emplace_back(std::move(mask));
            ++result.num;
        }

        return result;
    }

    // PoseModel 的后处理方法实现
    PoseRes postProcessPose(int idx) {
        auto& input_tensor = backend_->tensor_infos[0];
        auto& score_tensor = backend_->tensor_infos[1];
        auto& class_tensor = backend_->tensor_infos[2];
        auto& kpt_tensor   = backend_->tensor_infos[3];
        int   nkpt         = kpt_tensor.shape.d[2];
        int   ndim         = kpt_tensor.shape.d[3];

        int max_num = score_tensor.shape.d[1];
        auto label_at = [&](int i) -> int {
            switch (class_tensor.dtype()) {
                case nvinfer1::DataType::kINT32:
                    return static_cast<int*>(class_tensor.buffer->host())[idx * class_tensor.shape.d[1] + i];
                case nvinfer1::DataType::kINT64:
                    return static_cast<int>(static_cast<int64_t*>(class_tensor.buffer->host())[idx * class_tensor.shape.d[1] + i]);
                case nvinfer1::DataType::kFLOAT:
                    return static_cast<int>(static_cast<float*>(class_tensor.buffer->host())[idx * class_tensor.shape.d[1] + i]);
                default:
                    throw std::runtime_error(MAKE_ERROR_MESSAGE("PoseModel: unsupported labels dtype"));
            }
        };
        float* scores = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
        float* kpts   = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

        PoseRes result;
        result.num   = 0;
        int height   = input_tensor.shape.d[2];
        int width    = input_tensor.shape.d[3];
        float conf_thresh = backend_->infer_config.config.conf_thresh;

        auto& transform = backend_->infer_config.input_shape.has_value()
                              ? backend_->transforms.front()
                              : backend_->transforms[idx];

        result.boxes.reserve(max_num);
        result.scores.reserve(max_num);
        result.classes.reserve(max_num);
        result.kpts.reserve(max_num);

        for (int i = 0; i < max_num; ++i) {
            if (scores[i] <= conf_thresh) continue;
            std::vector<KeyPoint> keypoints;
            keypoints.reserve(nkpt);
            float left = 0.0f, top = 0.0f, right = 0.0f, bottom = 0.0f;
            for (int j = 0; j < nkpt; ++j) {
                float x = kpts[i * nkpt * ndim + j * ndim] * width;
                float y = kpts[i * nkpt * ndim + j * ndim + 1] * height;
                transform.apply(x, y, &x, &y);

                if (j == 0) {
                    left = right = x;
                    top = bottom = y;
                } else {
                    left   = std::min(left, x);
                    top    = std::min(top, y);
                    right  = std::max(right, x);
                    bottom = std::max(bottom, y);
                }
                keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
            }

            result.boxes.emplace_back(Box{left, top, right, bottom});
            result.scores.push_back(scores[i]);
            result.classes.push_back(label_at(i));
            result.kpts.emplace_back(std::move(keypoints));
            ++result.num;
        }
        return result;
    }

private:
    std::unique_ptr<TrtBackend> backend_;           // < TensorRT 后端
    unsigned long long          total_request_{0};  // < 总请求数
    std::unique_ptr<GpuTimer>   infer_gpu_trace_;   // < GPU推理计时器
    std::unique_ptr<CpuTimer>   infer_cpu_trace_;   // < CPU推理计时器
    DetectVariant               detect_variant_resolved_{DetectVariant::Auto};  // < 缓存的检测变体（Auto 表示尚未解析）
};

BaseModel::BaseModel()  = default;
BaseModel::~BaseModel() = default;

BaseModel::BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : impl_(std::make_unique<Impl>(trt_engine_file, infer_option)) {}

int BaseModel::batch() const {
    return impl_->batch();
}

std::tuple<std::string, std::string, std::string> BaseModel::performanceReport() {
    return impl_->performanceReport();
}

ClassifyModel::ClassifyModel()  = default;
ClassifyModel::~ClassifyModel() = default;

ClassifyModel::ClassifyModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<ClassifyModel> ClassifyModel::clone() const {
    auto clone_model   = std::make_unique<ClassifyModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<ClassifyRes> ClassifyModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<ClassifyRes> {
        std::vector<ClassifyRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessClassify(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<ClassifyRes>>(images, processImages);
}

ClassifyRes ClassifyModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

DetectModel::DetectModel()  = default;
DetectModel::~DetectModel() = default;

DetectModel::DetectModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<DetectModel> DetectModel::clone() const {
    auto clone_model   = std::make_unique<DetectModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<DetectRes> DetectModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<DetectRes> {
        std::vector<DetectRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessDetect(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<DetectRes>>(images, processImages);
}

DetectRes DetectModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

OBBModel::OBBModel()  = default;
OBBModel::~OBBModel() = default;

OBBModel::OBBModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<OBBModel> OBBModel::clone() const {
    auto clone_model   = std::make_unique<OBBModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<OBBRes> OBBModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<OBBRes> {
        std::vector<OBBRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessOBB(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<OBBRes>>(images, processImages);
}

OBBRes OBBModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

SegmentModel::SegmentModel()  = default;
SegmentModel::~SegmentModel() = default;

SegmentModel::SegmentModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<SegmentModel> SegmentModel::clone() const {
    auto clone_model   = std::make_unique<SegmentModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<SegmentRes> SegmentModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<SegmentRes> {
        std::vector<SegmentRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessSegment(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<SegmentRes>>(images, processImages);
}

SegmentRes SegmentModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

PoseModel::PoseModel()  = default;
PoseModel::~PoseModel() = default;

PoseModel::PoseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : BaseModel(trt_engine_file, infer_option) {}

std::unique_ptr<PoseModel> PoseModel::clone() const {
    auto clone_model   = std::make_unique<PoseModel>();
    clone_model->impl_ = impl_->clone();
    return clone_model;
}

std::vector<PoseRes> PoseModel::predict(const std::vector<Image>& images) {
    auto processImages = [this](size_t num) -> std::vector<PoseRes> {
        std::vector<PoseRes> results(num);
        for (size_t idx = 0; idx < num; ++idx) {
            results[idx] = this->impl_->postProcessPose(idx);
        }
        return results;
    };

    // withPerformanceReport
    return impl_->withPerformanceReport<decltype(processImages), std::vector<PoseRes>>(images, processImages);
}

PoseRes PoseModel::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

}  // namespace trtdetr
