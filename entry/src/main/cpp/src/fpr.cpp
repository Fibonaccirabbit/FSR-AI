//
// Created on 2024/4/16.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "mindsporeLite.h"
#include "nn.h"
#include "opencv2/imgproc/types_c.h"
#include "third_party/rknn/3rdparty/opencv/opencv-linux-aarch64/include/opencv2/imgproc.hpp"
#include "third_party/rknn/3rdparty/opencv/opencv-linux-armhf/include/opencv2/imgproc.hpp"
#include <semaphore.h>
#define GET_PARAMS(env, info, num)    \
    size_t argc = num;                \
    napi_value argv[num] = {nullptr}; \
    napi_value thisVar = nullptr;     \
    void *data = nullptr;             \
    napi_get_cb_info(env, info, &argc, argv, &thisVar, &data)

OH_AI_ModelHandle modelms = nullptr;
sem_t mutex;

std::string base64_encode(const unsigned char *input, int length) {
    std::string base64_start = "data:image/png;base64,";
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string output;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (length--) {
        char_array_3[i++] = *(input++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] =
                ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] =
                ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++)
                output += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] =
            ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] =
            ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++)
            output += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            output += '=';
    }

    return base64_start + output;
}

std::string matToBase64(cv::Mat &mat) {
    std::vector<uchar> buf;
    cv::imencode(".png", mat, buf);

    return base64_encode(buf.data(), buf.size());
}

std::string RunFPRModel(OH_AI_ModelHandle model, float *&imageData) {
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
        return "error";
    }
    LOGI("Run MSLite model success.\n");

    LOGI("Get model outputs:\n");
    for (size_t i = 0; i < outputs.handle_num; i++) {
        auto tensor = outputs.handle_list[i];
        LOGI("- TensorOutput %{public}d name is: %{public}s.\n", static_cast<int>(i), OH_AI_TensorGetName(tensor));
        LOGI("- TensorOutput %{public}d size is: %{public}d.\n", static_cast<int>(i), (int)OH_AI_TensorGetDataSize(tensor));
        auto out_data = reinterpret_cast<const float *>(OH_AI_TensorGetData(tensor));

        std::cout << "Output data is:";
        for (int i = 0; (i < OH_AI_TensorGetElementNum(tensor)) && (i <= 10); i++) {
            std::cout << out_data[i] << " ";
            LOGI("- TensorOutput %{public}d name is: %{public}f.\n", static_cast<int>(i), out_data[i]);
        }

        std::cout << std::endl;
    }
    auto tensor = outputs.handle_list[0];
    auto out_data = reinterpret_cast<const float *>(OH_AI_TensorGetData(tensor));
    cv::Mat hsi;
    hsi = cv::Mat(224, 224, CV_32FC(31), const_cast<float *>(out_data)).clone();
   
    std::vector<cv::Mat> img_channel(31);
    cv::split(hsi, img_channel);
    cv::normalize(hsi, hsi, 0, 1, cv::NORM_MINMAX);
    cv::Mat uint8Image;
    img_channel[30].convertTo(uint8Image, CV_8U, 255.0);
    cv::Mat color_image;
    cv::applyColorMap(uint8Image, color_image, cv::COLORMAP_JET);
    std::string result_str = matToBase64(color_image);
    return result_str;
}

static napi_value modelInit(napi_env env, napi_callback_info info) {
    // int32_t ret;
    int32_t modelId;
    napi_value error_ret;
    napi_create_int32(env, -1, &error_ret);
    napi_value success_ret;

    napi_value result = nullptr;
    napi_get_undefined(env, &result);

    GET_PARAMS(env, info, 1);
    const std::string modelName = "FPR-oh.ms";
    size_t modelSize;
    // 加载模型
    auto resourcesManager = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    auto modelBuffer = ReadModelFile(resourcesManager, modelName, &modelSize);
    modelId = 0;
    if (modelBuffer == nullptr) {
        LOGE("Read model failed");
        modelId = -1;
        return error_ret;
    }
    LOGI("Read model file success");

    modelms = CreateMSLiteModel(modelBuffer, modelSize);
    DestroyModelBuffer(&modelBuffer);
    if (modelms == nullptr) {
        modelId = -1;
        LOGE("MSLiteFwk Build model failed.\n");
        return error_ret;
    }
    LOGI("Create model file success");

    napi_create_int32(env, modelId, &success_ret);
    LOGI("ModelInit");

    return success_ret;
}
static napi_value modelDeInit(napi_env env, napi_callback_info info) {
    LOGI("ObjectDectionDeinit");
    if (modelms != nullptr) {
        OH_AI_ModelDestroy(&modelms);
        modelms = nullptr;
    }
    sem_init(&mutex, 0, 1);
    napi_value success_ret;
    napi_create_int32(env, 0, &success_ret);
    return success_ret;
}
static napi_value modelInference(napi_env env, napi_callback_info info) {
    LOGI("fpr in it");
    napi_value error_ret;
    napi_create_int32(env, -1, &error_ret);
    if (modelms == nullptr) {
        LOGE("No model are running!");
        return error_ret;
    }
    GET_PARAMS(env, info, 1);
    // 读取北向ArrayBuffer
    size_t byteLength = 0;
    void *d = nullptr;
    napi_get_arraybuffer_info(env, argv[0], &d, &byteLength);
    // YUV转RGB
    cv::Mat rgbImage;
    cv::Mat yuv(480 * 3 / 2, 640, CV_8UC1);
    yuv.data = (unsigned char *)(d);
    cv::cvtColor(yuv, rgbImage, cv::COLOR_YUV420sp2RGB);

    // Resize
    cv::Size targetSize(224, 224);
    cv::resize(rgbImage, rgbImage, targetSize);
    rgbImage.convertTo(rgbImage, CV_32FC3);

    // Norm to (0,1)
    cv::normalize(rgbImage, rgbImage, 0, 1.0, cv::NORM_MINMAX);
    
//    cv::findContours(rgbImage,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);
    
    // copy Mat.data
    float *inputData = new float[224 * 224 * 3];
    std::memcpy(inputData, rgbImage.data, 224 * 224 * 3 * sizeof(float));

    std::string res = RunFPRModel(modelms, inputData);
    napi_value result;
    const char *constc = nullptr; // 初始化const char*类型，并赋值为空
    constc = res.c_str();         // string类型转const char*类型
    napi_create_string_utf8(env, constc, res.size(), &result);

    LOGI("Exit runDemo()");
    return result;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"modelInit", nullptr, modelInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"modelDeInit", nullptr, modelDeInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"modelInference", nullptr, modelInference, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module fprModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "fpr",
    .nm_priv = ((void *)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterFPRModule(void) {
    napi_module_register(&fprModule);
}