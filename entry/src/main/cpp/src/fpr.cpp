//
// Created on 2024/4/16.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "mindsporeLite.h"
#include "nn.h"
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



void RunFPRModel(OH_AI_ModelHandle model, float *&imageData , float *& output) {
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
    }
    LOGI("Run MSLite model success.\n");

    LOGI("Get model outputs:\n");
    int ret = 0;
    auto tensor = outputs.handle_list[0];
    auto out_data = reinterpret_cast<const float *>(OH_AI_TensorGetData(tensor));
    float *input = new float[224*224*3];
    memcpy(input, out_data, 224*224*3*sizeof(float));
    cv::Mat image(224, 224, CV_32FC(3), input);
    LOGI("output colormap:%{public}d",image.rows*image.cols*image.elemSize());
    cv::Mat rgbaImage;
    cv::cvtColor(image, rgbaImage, cv::COLOR_RGB2RGBA);
//    std::vector<cv::Mait> channels;
//    cv::split(image,channels);
//    cv::Mat colorImage;
//    cv::applyColorMap(channels[0], colorImage , cv::COLORMAP_JET);
  
    memcpy(output,rgbaImage.data,224*224*4*sizeof(float));

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
    float *inputData = static_cast<float *>(d);
    LOGI("长度：%{public}zu", byteLength);
    nn::transform::Resizer resizer(374,500, 224, 224);
    resizer.resize(inputData);
    nn::Config::MobilenetV2Config mConfig(false, true, true);
    nn::PreProcessor PP(224, 224, mConfig);
    PP.RGBA = true;
    PP.Norm = true;
    PP.HWC = false;
    // 进行前处理
    PP.call(inputData);
    float * output = new float[224*224*4];
    RunFPRModel(modelms, inputData,output);
    napi_value result;
    void * outdata = (void *)output;
    napi_create_arraybuffer(env,224*224*4*sizeof(float),&outdata,&result);

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