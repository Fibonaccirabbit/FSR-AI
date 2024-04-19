//
// Created on 2024/4/16.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef ai_data_type_H
#define ai_data_type_H
#include "stdio.h"
#include <string.h>
#include <string>

const int OBJ_NUMB_MAX_SIZE = 100;
struct PicDesc {
    size_t width = 0;
    size_t height = 0;
    size_t dataSize = 0;
};

struct ObjectDesc {
    float left = 0;
    float top = 0;
    float right = 0;
    float bottom = 0;
    float prop = 0.0;
    std::string name = "";   
};
struct SSDInferResult {
    ObjectDesc objects[OBJ_NUMB_MAX_SIZE] = {};
    int count = -1;
};

#endif //ai_data_type_H
