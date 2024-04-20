////
//// Created on 2024/4/20.
////
//// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
//// please include "napi/native_api.h".
//
//#include "napi/native_api.h"
//#include "mindsporeLite.h"
//#include "nn.h"
//#include "opencv2/tracking.hpp"
//#define GET_PARAMS(env, info, num)    \
//    size_t argc = num;                \
//    napi_value argv[num] = {nullptr}; \
//    napi_value thisVar = nullptr;     \
//    void *data = nullptr;             \
//    napi_get_cb_info(env, info, &argc, argv, &thisVar, &data)
//
//cv::Mat gray, prev_gray;
//cv::Mat flow;
//static napi_value eyeTracking(napi_env env, napi_callback_info info) {
//
//    napi_value error_ret;
//    napi_create_int32(env, -1, &error_ret);
//    napi_value success_ret;
//
//    GET_PARAMS(env, info, 1);
//    // 读取北向ArrayBuffer
//    size_t byteLength = 0;
//    void *buffer = nullptr;
//    napi_get_arraybuffer_info(env, argv[0], &buffer, &byteLength);
//    cv::Mat rgbImage;
//    cv::Mat yuv(480 * 3 / 2, 640, CV_8UC1);
//    yuv.data = (unsigned char *)(buffer);
//    cv::cvtColor(yuv, rgbImage, cv::COLOR_YUV420sp2BGR);
//
//    double stop_threshold = 1.0;
//    cv::cvtColor(rgbImage, gray, cv::COLOR_BGR2RGB);
//
//    if (prev_gray.empty()) {
//        gray.copyTo(prev_gray);
//    };
//
//    cv::calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//    prev_gray = gray.clone();
//    double mean_flow_x = cv::mean(flow)[0];
//    double mean_flow_y = cv::mean(flow)[1];
//
//    if (std::abs(mean_flow_x) > stop_threshold || std::abs(mean_flow_y) > stop_threshold) {
//        if (std::abs(mean_flow_x) > std::abs(mean_flow_y)) {
//            if (mean_flow_x > 0) {
//                std::cout << "向右移动" << std::endl;
//            } else {
//                std::cout << "向左移动" << std::endl;
//            }
//        } else {
//            if (mean_flow_y > 0) {
//                std::cout << "向下移动" << std::endl;
//            } else {
//                std::cout << "向上移动" << std::endl;
//            }
//        }
//    } else {
//        std::cout << "停止注视" << std::endl;
//    }
//
//    napi_create_int32(env, 1, &success_ret);
//    LOGI("eyeTracking successfully!");
//
//    return success_ret;
//}
//
//EXTERN_C_START
//static napi_value Init(napi_env env, napi_value exports) {
//    napi_property_descriptor desc[] = {
//        {"eyeTracking", nullptr, eyeTracking, nullptr, nullptr, nullptr, napi_default, nullptr},
//        //        {"modelDeInit", nullptr, modelDeInit, nullptr, nullptr, nullptr, napi_default, nullptr},
//        //        {"modelInference", nullptr, modelInference, nullptr, nullptr, nullptr, napi_default, nullptr},
//    };
//    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
//    return exports;
//}
//EXTERN_C_END
//
//static napi_module eyeTrackingModule = {
//    .nm_version = 1,
//    .nm_flags = 0,
//    .nm_filename = nullptr,
//    .nm_register_func = Init,
//    .nm_modname = "eyetracking",
//    .nm_priv = ((void *)0),
//    .reserved = {0},
//};
//
//extern "C" __attribute__((constructor)) void RegisterEyeTrackingModule(void) {
//    napi_module_register(&eyeTrackingModule);
//}
