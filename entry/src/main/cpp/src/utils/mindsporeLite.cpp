//
// Created on 2024/4/15.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "mindsporeLite.h"
#include "nn.h"


cv::Mat letterbox(cv::Mat &src, int h, int w, std::vector<float> &pad) {

    int in_w = src.cols; // width
    int in_h = src.rows; // height
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int pad_w = tar_w - inside_w;
    int pad_h = tar_h - inside_h;

    cv::Mat resize_img;

    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

    pad_w = pad_w / 2;
    pad_h = pad_h / 2;

    pad.push_back(pad_w);
    pad.push_back(pad_h);
    pad.push_back(r);

    int top = int(round(pad_h - 0.1));
    int bottom = int(round(pad_h + 0.1));
    int left = int(round(pad_w - 0.1));
    int right = int(round(pad_w + 0.1));
    cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

    return resize_img;
}

int RunMSLiteModel(OH_AI_ModelHandle model, float *imageData) {
    // 设置模型输入数据
    auto inputs = OH_AI_ModelGetInputs(model);
    LOGI("Get model inputs:\n");

    for (size_t i = 0; i < 10; ++i) {
        LOGI("RunMSLiteModel imageData:%f", imageData[i]);
    }
    OH_AI_TensorSetData(inputs.handle_list[0], imageData);

    for (size_t i = 0; i < inputs.handle_num; i++) {
        auto tensor = inputs.handle_list[i];
        LOGI("- TensorInput %{public}d name is: %{public}s.\n", static_cast<int>(i), OH_AI_TensorGetName(tensor));
        LOGI("- TensorInput %{public}d size is: %{public}d.\n", static_cast<int>(i), (int)OH_AI_TensorGetDataSize(tensor));
        auto in_data = reinterpret_cast<const float *>(OH_AI_TensorGetData(tensor));
        std::cout << "Input data is:";
        for (int i = 0; (i < OH_AI_TensorGetElementNum(tensor)) && (i <= 10); i++) {
            std::cout << in_data[i] << " ";
            LOGI("- TensorInput %{public}d name is: %{public}f.\n", static_cast<int>(i), in_data[i]);
        }
        std::cout << std::endl;
    }

    auto outputs = OH_AI_ModelGetOutputs(model);

    // 执行推理并打印输出
    auto predict_ret = OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
    if (predict_ret != OH_AI_STATUS_SUCCESS) {
        OH_AI_ModelDestroy(&model);
        LOGE("Predict MSLite model error.\n");
        return -1;
    }
    LOGI("Run MSLite model success.\n");

    LOGI("Get model outputs:\n");
    int ret = 0;
    for (size_t i = 0; i < outputs.handle_num; i++) {
        auto tensor = outputs.handle_list[i];
        LOGI("- TensorOutput %{public}d name is: %{public}s.\n", static_cast<int>(i), OH_AI_TensorGetName(tensor));
        LOGI("- TensorOutput %{public}d size is: %{public}d.\n", static_cast<int>(i), (int)OH_AI_TensorGetDataSize(tensor));
        auto out_data = reinterpret_cast<const float *>(OH_AI_TensorGetData(tensor));
        
        nn::Classifier classifier(1001);
        classifier.need_softmax = true;
        ret = classifier.call(out_data);
        
        LOGI("- TensorOutput class is: %{public}d.\n", ret);

        std::cout << "Output data is:";
        for (int i = 0; (i < OH_AI_TensorGetElementNum(tensor)) && (i <= 10); i++) {
            std::cout << out_data[i] << " ";
            LOGI("- TensorOutput %{public}d name is: %{public}f.\n", static_cast<int>(i), out_data[i]);
        }

        std::cout << std::endl;
    }
//    OH_AI_ModelDestroy(&model);
    return ret;
}

void *ReadModelFile(NativeResourceManager *nativeResourceManager, const std::string &modelName, size_t *modelSize) {
    auto rawFile = OH_ResourceManager_OpenRawFile(nativeResourceManager, modelName.c_str());
    if (rawFile == nullptr) {
        LOGI("Open model file failed");
        return nullptr;
    }
    long fileSize = OH_ResourceManager_GetRawFileSize(rawFile);
    void *modelBuffer = malloc(fileSize);
    if (modelBuffer == nullptr) {
        LOGI("Get model file size failed");
    }
    int ret = OH_ResourceManager_ReadRawFile(rawFile, modelBuffer, fileSize);
    if (ret == 0) {
        LOGI("Read model file failed");
        OH_ResourceManager_CloseRawFile(rawFile);
        return nullptr;
    }
    OH_ResourceManager_CloseRawFile(rawFile);
    *modelSize = fileSize;
    return modelBuffer;
}

void DestroyModelBuffer(void **buffer) {
    if (buffer == nullptr) {
        return;
    }
    free(*buffer);
    *buffer = nullptr;
}

OH_AI_ModelHandle CreateMSLiteModel(void *modelBuffer, size_t modelSize) {
    // 创建上下文
    auto context = OH_AI_ContextCreate();
    LOGI("Creat Context.\n");
    if (context == nullptr) {
        LOGE("Create MSLite context failed.\n");
        return nullptr;
    }
    
    auto nnrt_device_info = OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_ACCELERATOR);
    OH_AI_ContextAddDeviceInfo(context, nnrt_device_info);
    auto cpu_device_info = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    OH_AI_ContextAddDeviceInfo(context, cpu_device_info);

    // 加载.ms模型文件
    auto model = OH_AI_ModelCreate();
    if (model == nullptr) {
        LOGE("Allocate MSLite Model failed.\n");
        return nullptr;
    }

    auto build_ret = OH_AI_ModelBuild(model, modelBuffer, modelSize, OH_AI_MODELTYPE_MINDIR, context);
    LOGI("Loader msfile.\n");
    if (build_ret != OH_AI_STATUS_SUCCESS) {
        OH_AI_ModelDestroy(&model);
        LOGE("Build MSLite model failed.\n");
        return nullptr;
    }
    LOGI("Build MSLite model success.\n");
    return model;
}