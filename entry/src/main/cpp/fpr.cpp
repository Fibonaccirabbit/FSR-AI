//
// Created on 2024/4/16.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#include "data_type.h"
#include "napi/native_api.h"
#include "mindsporeLite.h"
#include "nn.h"
#include "ssd_util.h"
#include <semaphore.h>
#define GET_PARAMS(env, info, num)    \
    size_t argc = num;                \
    napi_value argv[num] = {nullptr}; \
    napi_value thisVar = nullptr;     \
    void *data = nullptr;             \
    napi_get_cb_info(env, info, &argc, argv, &thisVar, &data)

OH_AI_ModelHandle modelms = nullptr;
sem_t mutex;


void RunSSD300Model(OH_AI_ModelHandle model, float *&imageData , SSDInferResult &result) {
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
    auto output1 = outputs.handle_list[0];
    auto branchScores = reinterpret_cast<const float *>(OH_AI_TensorGetData(output1));
    auto output2 = outputs.handle_list[1];
    auto branchBoxData = reinterpret_cast<const float *>(OH_AI_TensorGetData(output2));
    SSDModelUtil ssdmodelutil(1280,960);
    ssdmodelutil.getDecodeResult(branchScores,branchBoxData,result);
}
napi_status SetInferResult(napi_env env, napi_value result, SSDInferResult& inferResult) {
    napi_value objects = nullptr;
    napi_create_array_with_length(env, inferResult.count, &objects);
    for (int i = 0; i < inferResult.count; i++) {
        napi_value object = nullptr;
        napi_status status = napi_create_object(env, &object);
        if (status != napi_ok) {
            LOGE("napi_create_object failed.");
            return status;
        }
        napi_value left  = nullptr;
        LOGI("left %{}f",inferResult.objects[i].left);
        status = napi_create_double(env, inferResult.objects[i].left, &left);
        status = napi_set_named_property(env, object, "left", left);

        napi_value top  = nullptr;
        status = napi_create_double(env, inferResult.objects[i].top, &top);
        status = napi_set_named_property(env, object, "top", top);

        napi_value right  = nullptr;
        status = napi_create_double(env, inferResult.objects[i].right, &right);
        status = napi_set_named_property(env, object, "right", right);

        napi_value bottom  = nullptr;
        status = napi_create_double(env, inferResult.objects[i].bottom, &bottom);
        status = napi_set_named_property(env, object, "bottom", bottom);

        napi_value prop  = nullptr;
        status = napi_create_double(env, inferResult.objects[i].prop, &prop);
        status = napi_set_named_property(env, object, "prop", prop);

        napi_value name  = nullptr;
        status = napi_create_string_utf8(env, inferResult.objects[i].name.c_str(), NAPI_AUTO_LENGTH, &name);
        status = napi_set_named_property(env, object, "name", name);

        if (status != napi_ok) {
            LOGE("napi_create_xxx or napi_set_named_property failed. code: %{public}d", status);
            return status;
        }
 
        status = napi_set_element(env, objects, i, object);
        if (status != napi_ok) {
            LOGE("napi_set_element failed. code: %{public}d", status);
            return status;
        }
    }

    napi_status status = napi_set_named_property(env, result, "objects", objects);
    if (status != napi_ok) {
        LOGE("napi_set_named_property failed. code: %{public}d", status);
        return status;
    }

    napi_value count  = nullptr;
    status = napi_create_int32(env, inferResult.count, &count);
    status = napi_set_named_property(env, result, "count", count);
    if (status != napi_ok) {
        LOGE("napi_create_int32 or napi_set_named_property failed. code: %{public}d", status);
        return status;
    }

    return napi_ok;
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
    const std::string modelName = "ssd.ms";
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
//    sem_wait(&mutex);
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
    nn::transform::Resizer resizer(480,640, 300, 300);
    cv::Vec3d mean(0.485, 0.456, 0.406);
    cv::Vec3d var(1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225);
    resizer.resizeWithNorm(inputData,mean,var);
    nn::Config::MobilenetV2Config mConfig(false, false, true);
    nn::PreProcessor PP(300, 300, mConfig);
    PP.RGBA = true;
    PP.Norm = false;
    PP.HWC = false;
    // 进行前处理
    PP.call(inputData);
    SSDInferResult ssdinferresult;
    RunSSD300Model(modelms, inputData,ssdinferresult);
    napi_value success_ret;
    napi_create_int32(env,1, &success_ret);
    
    napi_value result = nullptr;
    napi_status status = napi_create_object(env, &result);
    status = SetInferResult(env,result,ssdinferresult);
//    sem_post(&mutex);
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

static napi_module ssd300Module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "ssd300",
    .nm_priv = ((void *)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterSSD300Module(void) {
    napi_module_register(&ssd300Module);
}
