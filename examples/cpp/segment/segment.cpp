/**
 * @file segment.cpp
 * @brief Segment C++ 示例
 * @date 2025-06-07
 *
 */
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "trtdetr.hpp"

namespace fs = std::filesystem;

// 获取指定目录中的图像文件
std::vector<std::string> get_images_in_directory(const std::string& folder_path) {
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        const auto extension = entry.path().extension().string();
        if (fs::is_regular_file(entry) && (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp")) {
            image_files.push_back(entry.path().string());
        }
    }
    return image_files;
}

// 创建输出目录
void create_output_directory(const std::string& output_path) {
    if (!fs::exists(output_path) && !fs::create_directories(output_path)) {
        throw std::runtime_error("Failed to create output directory: " + output_path);
    } else if (!fs::is_directory(output_path)) {
        throw std::runtime_error("Output path exists but is not a directory: " + output_path);
    }
}

// 从文件中生成标签
std::vector<std::string> generate_labels(const std::string& label_file) {
    std::ifstream file(label_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + label_file);
    }

    std::vector<std::string> labels;
    std::string              label;
    while (std::getline(file, label)) {
        labels.emplace_back(label);
    }
    return labels;
}

// 可视化推理结果（分割任务）
void visualize(cv::Mat& image, trtdetr::SegmentRes& result, const std::vector<std::string>& labels) {
    int im_h = image.rows;  // 图像高度
    int im_w = image.cols;  // 图像宽度

    // 遍历每个检测到的目标
    for (size_t i = 0; i < result.num; ++i) {
        auto&       box        = result.boxes[i];                          // 当前目标的边界框
        int         cls        = result.classes[i];                        // 当前目标的类别
        float       score      = result.scores[i];                         // 当前目标的置信度
        auto&       label      = labels[cls];                              // 获取类别对应的标签
        std::string label_text = label + " " + cv::format("%.3f", score);  // 构造显示的标签文本
        if (cls != 0) {
            continue;
        }
        // 绘制边界框和标签
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);

        // DETR 分割 mask 是模型输入空间的低分辨率整图 mask，不是 bbox-local mask。
        // 这里根据 letterbox 规则先裁掉 padding 区域，再缩放回原图大小。
        cv::Mat model_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());

        int   model_w = result.masks[i].width * 4;
        int   model_h = result.masks[i].height * 4;
        float scale   = std::min(static_cast<float>(model_w) / im_w, static_cast<float>(model_h) / im_h);
        int   valid_w = static_cast<int>(std::round(scale * im_w));
        int   valid_h = static_cast<int>(std::round(scale * im_h));
        int   offset_x = (model_w - valid_w) / 2;
        int   offset_y = (model_h - valid_h) / 2;

        int crop_x = static_cast<int>(std::round(offset_x * static_cast<float>(result.masks[i].width) / model_w));
        int crop_y = static_cast<int>(std::round(offset_y * static_cast<float>(result.masks[i].height) / model_h));
        int crop_w = static_cast<int>(std::round(valid_w * static_cast<float>(result.masks[i].width) / model_w));
        int crop_h = static_cast<int>(std::round(valid_h * static_cast<float>(result.masks[i].height) / model_h));

        crop_x = std::max(0, std::min(crop_x, model_mask.cols - 1));
        crop_y = std::max(0, std::min(crop_y, model_mask.rows - 1));
        crop_w = std::max(1, std::min(crop_w, model_mask.cols - crop_x));
        crop_h = std::max(1, std::min(crop_h, model_mask.rows - crop_y));

        cv::Mat valid_mask = model_mask(cv::Rect(crop_x, crop_y, crop_w, crop_h));
        cv::Mat float_mask;
        cv::resize(valid_mask, float_mask, image.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat mask_image;
        cv::threshold(float_mask, mask_image, 0.5, 255, cv::THRESH_BINARY);
        mask_image.convertTo(mask_image, CV_8UC1);

        // 创建一个与原图大小相同的颜色图像
        cv::Mat color_image(image.size(), image.type(), cv::Scalar(251, 81, 163));

        // 使用掩码将颜色图像与原图进行混合
        cv::Mat masked_color_image;
        cv::bitwise_and(color_image, color_image, masked_color_image, mask_image);

        cv::addWeighted(image, 1.0, masked_color_image, 0.5, 0, image);
    }
}

// 解析命令行参数
void parse_arguments(int argc, char** argv, std::string& engine_path, std::string& input_path, std::string& output_path, std::string& label_path) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " -e <engine> -i <input> [-o <output>] [-l <labels>]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-e" || arg == "--engine") {
            engine_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            input_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            output_path = argv[++i];
        } else if (arg == "-l" || arg == "--labels") {
            label_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

// 处理单张图像
void process_single_image(const std::string& image_path, const std::string& output_path, trtdetr::SegmentModel& model, const std::vector<std::string>& labels) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image from path: " + image_path);
    }

    trtdetr::Image img(image.data, image.cols, image.rows);
    auto           result = model.predict(img);

    if (!output_path.empty()) {
        visualize(image, result, labels);
        fs::path output_file_path = output_path / fs::path(image_path).filename();
        cv::imwrite(output_file_path.string(), image);
    }
}

// 处理一批图像
void process_batch_images(const std::vector<std::string>& image_paths, const std::string& output_path, trtdetr::SegmentModel& model, const std::vector<std::string>& labels) {
    const int batch_size = model.batch();
    for (size_t i = 0; i < image_paths.size(); i += batch_size) {
        std::vector<cv::Mat>        images;
        std::vector<trtdetr::Image> img_batch;
        std::vector<std::string>    img_name_batch;

        for (size_t j = i; j < i + batch_size && j < image_paths.size(); ++j) {
            cv::Mat image = cv::imread(image_paths[j], cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("Failed to read image from path: " + image_paths[j]);
            }
            images.push_back(image);
            img_batch.emplace_back(image.data, image.cols, image.rows);
            img_name_batch.push_back(fs::path(image_paths[j]).filename().string());
        }

        auto results = model.predict(img_batch);

        if (!output_path.empty()) {
            for (size_t j = 0; j < images.size(); ++j) {
                visualize(images[j], results[j], labels);
                fs::path output_file_path = output_path + "/" + img_name_batch[j];
                cv::imwrite(output_file_path.string(), images[j]);
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        std::string engine_path, input_path, output_path, label_path;
        parse_arguments(argc, argv, engine_path, input_path, output_path, label_path);

        if (!fs::exists(engine_path)) {
            throw std::runtime_error("Engine path does not exist: " + engine_path);
        }
        if (!fs::exists(input_path) || (!fs::is_regular_file(input_path) && !fs::is_directory(input_path))) {
            throw std::runtime_error("Input path does not exist or is not a regular file/directory: " + input_path);
        }

        std::vector<std::string> labels;
        if (!output_path.empty()) {
            if (label_path.empty()) {
                throw std::runtime_error("Please provide a labels file using -l or --labels.");
            }
            if (!fs::exists(label_path)) {
                throw std::runtime_error("Label path does not exist: " + label_path);
            }
            labels = generate_labels(label_path);
            create_output_directory(output_path);
        }

        trtdetr::InferOption option;
        option.enableSwapRB();

        if (!fs::is_regular_file(input_path)) {
            option.enablePerformanceReport();
        }

        auto model = std::make_unique<trtdetr::SegmentModel>(engine_path, option);

        if (fs::is_regular_file(input_path)) {
            process_single_image(input_path, output_path, *model, labels);
        } else {
            auto image_files = get_images_in_directory(input_path);
            if (image_files.empty()) {
                throw std::runtime_error("Failed to read image from path: " + input_path);
            }
            process_batch_images(image_files, output_path, *model, labels);
        }

        std::cout << "Inference completed." << std::endl;

        if (!fs::is_regular_file(input_path)) {
            auto [throughput_str, gpu_latency_str, cpu_latency_str] = model->performanceReport();
            std::cout << throughput_str << std::endl;
            std::cout << gpu_latency_str << std::endl;
            std::cout << cpu_latency_str << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
