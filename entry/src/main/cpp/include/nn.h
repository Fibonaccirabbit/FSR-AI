//
// Created on 2024/4/15.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef ai_nn_H
#define ai_nn_H
#include "third_party/rknn/3rdparty/opencv/opencv-linux-armhf/include/opencv2/calib3d.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
namespace nn {
namespace Config {
class modelConfig {
  public:
    bool RGBA = true;
    bool HWC2CHW = false;
    bool shouldNorm = true;
    modelConfig(){

    };
    modelConfig(bool hwc2chw, bool norm) {
        HWC2CHW = hwc2chw;
        shouldNorm = norm;
    };
};
class MobilenetV2Config : modelConfig {
  public:
    bool HWC2CHW = false;
    bool RGBA = true;
    bool shouldNorm = true;
    MobilenetV2Config(bool hwc2chw, bool norm, bool rgba) {
        HWC2CHW = hwc2chw;
        shouldNorm = norm;
        RGBA = rgba;
    };
};

} // namespace Config
class PreProcessor {
    /********
     * @brief
     * 图像前处理器，提供前处理服务
     * @param
     * s_: 图像大小，仅需提供一个参数，长宽相等
     *********/

  public:
    int height = 224;
    int width = 224;
    bool RGBA = false; // 如果需要将 RGB 转为 RGBA ，设为true
    bool Norm = false; // 如果需要归一化，设为true
    bool soloNorm = false;
    bool HWC = false; // 如果需要将 channel last 转为 channel first ，设为true

    PreProcessor(int h_, int w_, nn::Config::MobilenetV2Config config) {
        height = h_;
        width = w_;
        RGBA = config.RGBA;
        soloNorm = config.shouldNorm;
        HWC = config.HWC2CHW;
    };
    void call(float *&image); // 调用此进行前处理
};
class Classifier {
    /********
     * @brief
     * 模型输出处理器，提供前处理及后处理服务
     * @param
     * msTensor: 图像的推理结果
     * n_: 类别数，适用于分类任务
     *********/

  public:
    std::string task = "classifier";
    int num_classes;
    bool need_softmax = false; // set true to use softmax
    Classifier(int n_ = 1001) { num_classes = n_; }
    int call(const float *result);
};

namespace method {
int Classification(const float *result, int &num_classes, bool &softmax);
}

namespace activateFun {
float *SoftMax(const float *tensor, int &num_classes);
}

namespace transform {
class Resizer {
  public:
    int cur_height;
    int cur_width;
    int target_height;
    int target_width;
    cv::Mat input;
    cv::Mat output;
    float *outputData;
    Resizer(int c_height, int c_width, int t_height, int t_width) {
        cur_height = c_height;
        cur_width = c_width;
        target_height = t_height;
        target_width = t_width;
    }
    void resize(float *&input);
    void resizeWithNorm(float *&input,cv::Vec3d mean,cv::Vec3d var);
    ~Resizer() {
    }
};

void RGBA2RGB(float *&inImage, int &height, int &width, bool &Norm);
float *HWC2CHW(float *&image, int &shape);
void Norm1(float *&image, int &height, int &width);
} // namespace transform

} // namespace nn
#endif // ai_nn_H
