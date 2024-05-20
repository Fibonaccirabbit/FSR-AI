//
// Created on 2024/4/15.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef ai_mindsporeLite_H
#define ai_mindsporeLite_H
#include <mindspore/model.h>
#include <mindspore/context.h>
#include <mindspore/status.h>
#include <iostream>
#include <mindspore/tensor.h>
#include <rawfile/raw_file_manager.h>
#include "Log.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat letterbox(cv::Mat &src, int h, int w, std::vector<float> &pad);
int RunMSLiteModel(OH_AI_ModelHandle model, float *imageData);
void *ReadModelFile(NativeResourceManager *nativeResourceManager, const std::string &modelName, size_t *modelSize);
void DestroyModelBuffer(void **buffer);
OH_AI_ModelHandle CreateMSLiteModel(void *modelBuffer, size_t modelSize);
#endif //ai_mindsporeLite_H
