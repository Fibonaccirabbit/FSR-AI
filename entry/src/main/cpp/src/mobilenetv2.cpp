//
// Created on 2024/4/15.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "napi/native_api.h"
#include "mindsporeLite.h"
#include "nn.h"
#include <semaphore.h>

#define GET_PARAMS(env, info, num)    \
    size_t argc = num;                \
    napi_value argv[num] = {nullptr}; \
    napi_value thisVar = nullptr;     \
    void *data = nullptr;             \
    napi_get_cb_info(env, info, &argc, argv, &thisVar, &data)

OH_AI_ModelHandle modelms = nullptr;
sem_t mutex;

static napi_value modelInit(napi_env env, napi_callback_info info) {
    // int32_t ret;
    int32_t modelId;
    napi_value error_ret;
    napi_create_int32(env, -1, &error_ret);
    napi_value success_ret;

    napi_value result = nullptr;
    napi_get_undefined(env, &result);

    GET_PARAMS(env, info, 1);
    const std::string modelName = "mobilenetv2.ms";
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
    sem_wait(&mutex);
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
    //    float *inputData = static_cast<float *>(d);
    //    LOGI("长度：%{public}zu", byteLength);
    //    nn::transform::Resizer resizer(480,640, 224, 224);
    //    resizer.resize(inputData);
    //    nn::Config::MobilenetV2Config mConfig(false, true, true);
    //    nn::PreProcessor PP(224, 224, mConfig);
    //    PP.RGBA = true;
    //    PP.Norm = true;
    //    PP.HWC = false;
    //    // 进行前处理
    //    PP.call(inputData);
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

    // copy Mat.data
    float *inputData = new float[224 * 224 * 3];
    std::memcpy(inputData, rgbImage.data, 224 * 224 * 3 * sizeof(float));

    int res = RunMSLiteModel(modelms, inputData);
    napi_value success_ret;
    napi_create_int32(env, res, &success_ret);
    sem_post(&mutex);
    LOGI("Exit runDemo()");
    return success_ret;
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

static napi_module mobilenetV2Module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "mobilenetV2",
    .nm_priv = ((void *)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterMobilenetV2Module(void) {
    napi_module_register(&mobilenetV2Module);
}
