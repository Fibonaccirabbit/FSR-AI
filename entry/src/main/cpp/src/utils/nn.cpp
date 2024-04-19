//
// Created on 2024/4/15.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#include "nn.h"
#include "Log.h"
#include "third_party/rknn/3rdparty/opencv/opencv-linux-armhf/include/opencv2/core.hpp"
namespace nn {

void nn::PreProcessor::call(float *&image) {
    if (RGBA)
        nn::transform::RGBA2RGB(image, height, width, Norm);
    if (soloNorm)
        nn::transform::Norm1(image, height, width);
    //    if (HWC) image = HWC2CHW(image, shape);
}
int nn::Classifier::call(const float *result) {
    if (task == "classifier") {
        return nn::method::Classification(result, num_classes, need_softmax);
    }
}

namespace method {
/********
 * @brief
 * 处理图像分类的结果，返回类别，-1代表未检测出任何类别
 * @param
 * result: Result of deeplearning model with shape (1, num_classes)
 * num_classes: Number of classification model output.
 * softmax: If need softmax, set true.
 *********/
int Classification(const float *result, int &num_classes, bool &softmax) {
    int ret = 0;
    float *temp = nullptr;

    if (softmax)
        temp = nn::activateFun::SoftMax(result, num_classes);

    for (int i = 0; i < num_classes; i++) {
        if (temp[i] > temp[ret])
            ret = i;
    }

    return ret;
}
} // namespace method

namespace activateFun {
float *SoftMax(const float *tensor, int &num_classes) {
    // Find maximum value in the tensor
    float temp[num_classes];
    float max_val = tensor[0];
    for (int i = 1; i < num_classes; ++i) {
        if (tensor[i] > max_val) {
            max_val = tensor[i];
        }
    }

    // Subtract maximum value for numerical stability
    float sum_exp = 0.0;
    for (int i = 0; i < num_classes; ++i) {
        temp[i] = exp(tensor[i]);
        sum_exp += temp[i];
    }

    // Normalize
    for (int i = 0; i < num_classes; ++i) {
        temp[i] /= sum_exp;
        //        LOGI("%{public}f",temp[i]);
    }

    return temp;
}
} // namespace activateFun

namespace transform {

void nn::transform::Resizer::resize(float *&inputData) {
    cv::Mat image(cur_height, cur_width, CV_32FC(4), inputData);
    // 创建一个目标尺寸
    cv::Size targetSize(target_height, target_width); // 新的宽度和高度
    // 创建一个用于存储缩放后图像数据的cv::Mat
    cv::Mat resizedImage(targetSize, CV_32FC(4));
    // 使用resize函数进行缩放
    cv::resize(image, resizedImage, targetSize);
    inputData = new float[target_height * target_width * 4];
    memcpy(inputData, resizedImage.data, target_height * target_width * 4 * sizeof(float));
    resizedImage.release();
    image.release();
}

void nn::transform::Resizer::resizeWithNorm(float *&inputData, cv::Vec3d mean, cv::Vec3d var){
        cv::Mat image(cur_height, cur_width, CV_32FC(4), inputData);
        // 创建一个目标尺寸
        cv::Size targetSize(target_height, target_width); // 新的宽度和高度
        // 创建一个用于存储缩放后图像数据的cv::Mat
        cv::Mat resizedImage(targetSize, CV_32FC(4));
        // 使用resize函数进行缩放
        cv::resize(image, resizedImage, targetSize);
        
        resizedImage.convertTo(resizedImage, CV_32FC(4),1.0 / 255.0);
        
        std::vector<cv::Mat> channels;
        cv::split(resizedImage,channels);
    
        for(int i =0;i<3;++i){
            cv::subtract(channels[i], mean[i], channels[i]);
            channels[i] /= var[i];
        }
        cv::merge(channels,resizedImage);
        inputData = new float[target_height * target_width * 4];
        memcpy(inputData, resizedImage.data, target_height * target_width * 4 * sizeof(float));
        LOGI("after resize:%{public}d",resizedImage.rows*resizedImage.cols*resizedImage.elemSize());
        resizedImage.release();
        image.release();
}

/********
 * @brief
 * 将图像进行归一化
 * @param
 * inImage: 需要处理的图像
 * shape: 图像的形状 （长宽需一致）
 *********/
void Norm1(float *&image, int &height, int &width) {
    int length = height * width * 3;
    for (int i = 0; i < length; i++)
        image[i] = image[i] / 255.0;
}

/********
 * @brief
 * 将 rgba rgba 图像转为 rgb rgb图像
 * @param
 * inImage: 需要处理的图像 （224 x 224 x 4）
 *********/
void RGBA2RGB(float *&inImage, int &height, int &width, bool &Norm) {
    int imageLength = height * width * 3;

    for (int i = 0; i < imageLength; i++)
        inImage[i] = inImage[i + i / 3];
}

/********
 * @brief
 * 将 HWC 图像转换为 CHW 图像
 * @param
 * image: 需要处理的图像 （shape x shape x 3）
 * shape: 图像大小, default=224
 * @return
 * CHW: 转换后的图像 (3 x shape x shape)
 *********/
float *HWC2CHW(float *&image, int &shape) {
    int imageLength = 3 * shape * shape;
    float *CHW = new float[imageLength];
    for (int i = 0; i < imageLength; i++) {
        int c = i % 3;
        int w = (i / 3) % shape;
        int h = i / 3 / shape;
        CHW[c * shape * shape + h * shape + w] = image[i];
    }
    return CHW;
}

} // namespace transform
} // namespace nn