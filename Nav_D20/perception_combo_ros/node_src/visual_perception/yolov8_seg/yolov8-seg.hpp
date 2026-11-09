//
// Created by ubuntu on 1/24/23.
//
#ifndef YOLOV8_SEG_HPP
#define YOLOV8_SEG_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>

using namespace seg;

// 定义分割结果结构体
struct DetectionResult {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
    cv::Mat mask;
};

// 定义输出结果结构体
struct ProcessResult {
    cv::Mat overlay_image;  // 叠加分割结果的图像
    std::vector<DetectionResult> detections;  // 检测结果列表
};

class YOLOv8_seg {
public:
    inline explicit YOLOv8_seg(const std::string& engine_file_path);
    inline ~YOLOv8_seg();

    inline void                 make_pipe(bool warmup = true);
    inline std::vector<std::string> getTensorNames(nvinfer1::TensorIOMode mode);
    inline void yolo_printDims(const nvinfer1::Dims& dims, const std::string& name);
    inline void                 check_output_data();
    inline void                 copy_from_Mat(const cv::Mat& image);
    inline void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    inline void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    inline void                 infer();
    inline void                 postprocess(std::vector<Object>& objs,
                                     float                score_thres  = 0.25f,
                                     float                iou_thres    = 0.65f,
                                     int                  topk         = 100,
                                     int                  seg_channels = 32,
                                     int                  seg_h        = 160,
                                     int                  seg_w        = 160);
    inline static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS,
                                      const std::vector<std::vector<unsigned int>>& MASK_COLORS,
                                      ProcessResult &outputResults);
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

void YOLOv8_seg::yolo_printDims(const nvinfer1::Dims& dims, const std::string& name = "Dims") {
    std::cout << name << ": [";
    for (int i = 0; i < dims.nbDims; ++i) {
        std::cout << dims.d[i];
        if (i < dims.nbDims - 1) {
            std::cout << " x ";
        }
    }
    std::cout << "]" << std::endl;
}
YOLOv8_seg::YOLOv8_seg(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;



    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbIOTensors();

    // step2  createInferenceTensors 每一个tensor都有一个name 是根据name来提取信息
    for (int i = 0; i < this->num_bindings; ++i) {

        const char* name = this->engine->getIOTensorName(i);
        std::cout << " CREATE IO " << name << std::endl;
        // std::string        name  = this->engine->getBindingName(i);

        nvinfer1::DataType dtype = this->engine->getTensorDataType(name);
        // nvinfer1::DataType dtype = this->engine->getBindingDataType(i);

        
        nvinfer1::Dims     dims;
        Binding            binding;
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        // bool IsInput = engine->bindingIsInput(i);
        bool IsInput = this->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        if (IsInput) {
            this->num_inputs += 1;
            // dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            dims = this->engine->getTensorShape(name);
            // std::cout << " CREATE IO  dims " << dims  << std::endl;
            yolo_printDims(dims," input ");
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape 动态维度 输入
            // this->context->setBindingDimensions(i, dims);
            this->context->setInputShape(name, dims);
        }
        else {
            // dims         = this->context->getBindingDimensions(i);
            dims = this->engine->getTensorShape(name);

            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            yolo_printDims(dims," output ");
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }



    }
}

YOLOv8_seg::~YOLOv8_seg()
{
    // trt10 不需要 destory？
    // this->context->destroy();
    // this->engine->destroy();
    // this->runtime->destroy();

    delete this->context;    // 使用 delete 而不是 context->destroy()
    delete this->engine;     // 使用 delete 而不是 engine->destroy()
    delete this->runtime;    // 使用 delete 而不是 runtime->destroy()


    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_seg::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);

                // trt10 新增的
                auto     width      = bindings.dims.d[3];
                auto     height     = bindings.dims.d[2];
                std::cout << "warm up Setting address for tensor: " << bindings.name << std::endl;
                
                this->context->setInputShape(bindings.name, nvinfer1::Dims{4, {1, 3, height, width}});

                // trt10 新增的
                this->context->setTensorAddress(bindings.name, this->device_ptrs[0]); 

                for (int i = 0; i < this->num_outputs; i++) {
                    auto& output_binding = this->output_bindings[i];
                    this->context->setTensorAddress(output_binding.name, this->device_ptrs[i + this->num_inputs]);
                }

                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8_seg::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
    // cv::imwrite("/home/huanghanlin/hhl_work/seres_work/dog_follow/D10_related/ros2_node/test_onnx_trt/test_letterbox.jpg",tmp);

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

std::vector<std::string> YOLOv8_seg::getTensorNames(nvinfer1::TensorIOMode mode){
    
    int totalIOTensors = this->engine->getNbIOTensors();
    std::vector<std::string> tensorNames;
    
    for(int i=0; i<totalIOTensors; i++){
        std::string name = this->engine->getIOTensorName(i);

        if(this->engine->getTensorIOMode(name.c_str()) == mode){
            tensorNames.push_back(std::move(name));
        }
    }
    
    return tensorNames;
}

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);
    

    // this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
    
    this->context->setInputShape(in_binding.name, nvinfer1::Dims{4, {1, 3, height, width}});

    // trt10 新增的
    // for (auto& bindings : this->output_bindings){
        this->context->setTensorAddress(in_binding.name, this->device_ptrs[0]); 
    // }
    

    // 设置输出张量地址（需要在每次推理前设置）
    for (int i = 0; i < this->num_outputs; i++) {
        auto& output_binding = this->output_bindings[i];
        this->context->setTensorAddress(output_binding.name, this->device_ptrs[i + this->num_inputs]);
    }

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    // this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});

    // trt10
    
    auto&    in_binding = this->input_bindings[0];

    std::cout << "Setting address for tensor: " << in_binding.name << std::endl;

    this->context->setInputShape(in_binding.name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});

    // trt10 新增的
    
    this->context->setTensorAddress(in_binding.name, this->device_ptrs[0]); 

    // 设置输出张量地址（需要在每次推理前设置）
    for (int i = 0; i < this->num_outputs; i++) {
        auto& output_binding = this->output_bindings[i];
        this->context->setTensorAddress(output_binding.name, this->device_ptrs[i + this->num_inputs]);
    }

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

    
}

void YOLOv8_seg::check_output_data()
{
    for (int i = 0; i < this->num_outputs; i++) {
        float* output_data = static_cast<float*>(this->host_ptrs[i]);
        size_t num_elements = this->output_bindings[i].size;
        
        std::cout << "Output " << i << " first 10 values: ";
        for (int j = 0; j < std::min(num_elements, size_t(10)); j++) {
            std::cout << output_data[j] << " ";
        }
        std::cout << std::endl;
        
        // 检查是否有有效数据（非零、非NaN）
        bool has_valid_data = false;
        for (size_t j = 0; j < num_elements; j++) {
            if (!std::isnan(output_data[j]) && output_data[j] != 0.0f) {
                has_valid_data = true;
                break;
            }
        }
        
        std::cout << "Output " << i << " has valid data: " << has_valid_data << std::endl;
    }
}
void YOLOv8_seg::infer()
{
    //trt10 新增的
    // for (int i = 0; i < this->num_outputs; i++) {
    //     auto&    output_binding = this->output_bindings[i];
    //     this->context->setTensorAddress(output_binding.name, this->device_ptrs[i + this->num_inputs]);
    // }
    

    // this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);

    // trt10
     this->context->enqueueV3(this->stream);

    for (int i = 0; i < this->num_outputs; i++) {

        auto& output_binding = this->output_bindings[i];
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);

    // check_output_data();
}

void YOLOv8_seg::postprocess(
    std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int seg_channels, int seg_h, int seg_w)
{
    objs.clear();
    auto input_h      = this->input_bindings[0].dims.d[2];
    auto input_w      = this->input_bindings[0].dims.d[3];
    auto num_anchors  = this->output_bindings[0].dims.d[1];
    auto num_channels = this->output_bindings[0].dims.d[2];
    std::cout << "  num_anchors  " <<num_anchors 
              << "num_channels " << num_channels << std::endl;

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    
    auto*   output = static_cast<float*>(this->host_ptrs[0]);
    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*>(this->host_ptrs[1]));
    
    std::vector<int>      labels;
    std::vector<float>    scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat>  mask_confs;
    std::vector<int>      indices;

    for (int i = 0; i < num_anchors; i++) {
        float* ptr   = output + i * num_channels;
        float  score = *(ptr + 4);
        if (score > score_thres) {
            float x0 = *ptr++ - dw;
            float y0 = *ptr++ - dh;
            float x1 = *ptr++ - dw;
            float y1 = *ptr++ - dh;
            // std::cout << " -- objs[i] x0  " <<  x0   
            //           << " -- objs[i] y0 " << y0
            //           << " -- objs[i] x1 "<< x1
            //           << " -- objs[i] y1 " << y1 << std::endl;      

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);

            // std::cout << " objs[i] x0  " <<  x0   
            //           << " objs[i] y0 " << y0
            //           << " objs[i] x1 "<< x1
            //           << " objs[i] y1 " << y1 << std::endl;

            // 确保x1 > x0, y1 > y0
            if (x1 <= x0 || y1 <= y0) {
                std::cout << "Invalid bbox coordinates, skipping: " 
                          << "x0=" << x0 << ", y0=" << y0 
                          << ", x1=" << x1 << ", y1=" << y1 << std::endl;
                continue;
            }

            int     label     = *(++ptr);
            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, ++ptr);
            mask_confs.push_back(mask_conf);
            labels.push_back(label);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));

            std::cout << " ========== labels " << label 
                      << " objs[i] x0  " <<  x0   
                      << " objs[i] y0 " << y0
                      << " objs[i] x1 "<< x1
                      << " objs[i] y1 " << y1 
                      << " score " <<score  << std::endl;
        }
    }

    // 分类别进行nms=====================================================================
    // === 修改这里：按类别分别进行NMS ===
    std::vector<std::vector<int>> class_indices;
    std::vector<int> all_indices;
    
    // 获取所有唯一的类别
    std::set<int> unique_labels(labels.begin(), labels.end());
    
    // 对每个类别分别进行NMS
    for (int class_id : unique_labels) {
        std::vector<cv::Rect> class_bboxes;
        std::vector<float> class_scores;
        std::vector<int> class_original_indices;
        
        // 收集当前类别的检测框
        for (int i = 0; i < bboxes.size(); i++) {
            if (labels[i] == class_id) {
                class_bboxes.push_back(bboxes[i]);
                class_scores.push_back(scores[i]);
                class_original_indices.push_back(i);
            }
        }
        
        if (class_bboxes.empty()) continue;
        
        std::vector<int> class_nms_indices;
#if defined(BATCHED_NMS)
        cv::dnn::NMSBoxesBatched(class_bboxes, class_scores, std::vector<int>(class_bboxes.size(), class_id), 
                                score_thres, iou_thres, class_nms_indices);
#else
        cv::dnn::NMSBoxes(class_bboxes, class_scores, score_thres, iou_thres, class_nms_indices);
#endif
        
        // 将原始索引添加到最终结果中
        for (int idx : class_nms_indices) {
            all_indices.push_back(class_original_indices[idx]);
        }
    }
    
    // 限制总数量
    if (all_indices.size() > topk) {
        all_indices.resize(topk);
    }
    
    // 使用 all_indices 代替原来的 indices
    cv::Mat masks;
    int cnt = 0;
    for (auto& i : all_indices) {  // 这里改为 all_indices
        if (cnt >= topk) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
        
        std::cout << " === After NMS === label: " << labels[i] 
                  << " score: " << scores[i] 
                  << " bbox: [" << tmp.x << ", " << tmp.y << ", " << tmp.width << ", " << tmp.height << "]" << std::endl;
    }
    // 分类别进行nms=====================================================================

// #if defined(BATCHED_NMS)
//     cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
// #else
//     cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
// #endif

//     cv::Mat masks;
//     int     cnt = 0;
//     for (auto& i : indices) {
//         if (cnt >= topk) {
//             break;
//         }
//         cv::Rect tmp = bboxes[i];
//         Object   obj;
//         obj.label = labels[i];
//         obj.rect  = tmp;
//         obj.prob  = scores[i];
//         masks.push_back(mask_confs[i]);
//         objs.push_back(obj);
//         cnt += 1;
//     }

    if (masks.empty()) {
        // masks is empty
        return ;
    }
    else {
        
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat   = matmulRes.reshape(all_indices.size(), {seg_h, seg_w});
        std::cout << " maskMat size " << maskMat.size() << std::endl;

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        // cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);
        int roi_width = seg_w - 2 * scale_dw;
        int roi_height = seg_h - 2 * scale_dh;
        
        if (roi_width <= 0 || roi_height <= 0) {
            std::cerr << "Invalid ROI, skipping mask processing" << std::endl;
            return;
        }
        cv::Rect roi(scale_dw, scale_dh, roi_width, roi_height);
        // std::cout << " indices.size() "<< indices.size() <<std::endl;
        for (int i = 0; i < all_indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);

            // 检查 dest 是否为空
            if (dest.empty()) {
                continue;
            }
            
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);

            std::cout << " objs[i].rect.x  " << objs[i].rect.x  
                      << " objs[i].rect.y " <<objs[i].rect.y
                      << " objs[i].rect.width "<<objs[i].rect.width 
                      << " objs[i].rect.height " <<objs[i].rect.height << std::endl;

            objs[i].boxMask = mask(objs[i].rect) > 0.5f;

            // cv::Mat gradientImage;
            // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            // cv::morphologyEx(dest, gradientImage, cv::MORPH_CLOSE, kernel);

            // cv::resize(gradientImage, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);

            // std::cout << " i "<< i <<std::endl;
            // objs[i].boxMask = mask(objs[i].rect) > 0.5f;

            


            // cv::Mat binary_mask;
            // cv::threshold(dest, binary_mask, 0.5f, 1.0f, cv::THRESH_BINARY);
            // cv::resize(binary_mask, mask, cv::Size((int)width, (int)height), cv::INTER_NEAREST);
            // objs[i].boxMask = mask(objs[i].rect);
        }
    }
}

void YOLOv8_seg::draw_objects(const cv::Mat&                                image,
                              cv::Mat&                                      res,
                              const std::vector<Object>&                    objs,
                              const std::vector<std::string>&               CLASS_NAMES,
                              const std::vector<std::vector<unsigned int>>& COLORS,
                              const std::vector<std::vector<unsigned int>>& MASK_COLORS,
                              ProcessResult &outputResults)
{
    res          = image.clone();
    cv::Mat mask = image.clone();
    outputResults.detections.clear();
    for (auto& obj : objs) {
        int        idx   = obj.label;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color =
            cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), obj.prob * 100);
        mask(obj.rect).setTo(mask_color, obj.boxMask);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

        // // 创建检测结果
        DetectionResult det_result;
        det_result.class_id = obj.label;
        det_result.class_name = CLASS_NAMES[idx];
        det_result.confidence = obj.prob;
        det_result.bbox = cv::Rect(x, y, label_size.width, label_size.height + baseLine);
        det_result.mask = obj.boxMask.clone();
        outputResults.detections.push_back(det_result);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
    outputResults.overlay_image = res;
}
#endif  // SEGMENT_SIMPLE_YOLOV8_SEG_HPP
