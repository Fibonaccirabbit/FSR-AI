# the minimum version of CMake.
cmake_minimum_required(VERSION 3.4.1)
project(ai)

set(CMAKE_CXX_STANDARD 17)

set(NATIVE_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${NATIVE_ROOT_PATH}
                    ${NATIVE_ROOT_PATH}/include
                    ${NATIVE_ROOT_PATH}/third_party/opencv/include/opencv4
                    ${NATIVE_ROOT_PATH}/third_party/rknn/runtime/Linux/librknn_api/include
                    )

add_library(depthanything SHARED
      src/depthanything.cpp
      src/utils/mindsporeLite.cpp
      src/utils/nn.cpp
    )

add_library(mobilenetV2 SHARED
      src/mobilenetv2.cpp
      src/utils/mindsporeLite.cpp
      src/utils/nn.cpp
    )
add_library(ssd300 SHARED
      src/ssd300.cpp
      src/utils/ssd_util.cpp
      src/utils/mindsporeLite.cpp
      src/utils/nn.cpp
    )
add_library(fpr SHARED
      src/fpr.cpp
      src/utils/mindsporeLite.cpp
      src/utils/nn.cpp
    )


add_library(rknnbenchmark SHARED
     src/rknnbenchmark.cpp
     src/utils/mindsporeLite.cpp
     src/utils/nn.cpp
)

# opencv
## libopencv_core
add_library( libopencv_core STATIC IMPORTED)
set_target_properties(libopencv_core
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_core.a)


## libopencv_imgcodecs
add_library( libopencv_imgcodecs STATIC IMPORTED)
set_target_properties(libopencv_imgcodecs
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_imgcodecs.a)


## libopencv_imgproc
add_library( libopencv_imgproc STATIC IMPORTED)
set_target_properties(libopencv_imgproc
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_imgproc.a)


## libopencv_highgui
add_library( libopencv_highgui STATIC IMPORTED)
set_target_properties(libopencv_highgui
       PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_highgui.a)


## libopencv_objdetect
add_library( libopencv_objdetect STATIC IMPORTED)
set_target_properties(libopencv_objdetect
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_objdetect.a)


## libopencv_calib3d
add_library( libopencv_calib3d STATIC IMPORTED)
set_target_properties(libopencv_calib3d
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_calib3d.a)


## libopencv_flann
add_library( libopencv_flann STATIC IMPORTED)
set_target_properties(libopencv_flann
       PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_flann.a)


## libopencv_features2d
add_library( libopencv_features2d STATIC IMPORTED)
set_target_properties(libopencv_features2d
       PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_features2d.a)


## libopencv_dnn
add_library( libopencv_dnn STATIC IMPORTED)
set_target_properties(libopencv_dnn
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_dnn.a)


## libopencv_videoio
add_library( libopencv_videoio STATIC IMPORTED)
set_target_properties(libopencv_videoio
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_videoio.a)


## libade
add_library( libade STATIC IMPORTED )
set_target_properties(libade
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/libade.a)


## libittnotify
add_library( libittnotify STATIC IMPORTED )
set_target_properties(libittnotify
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/libittnotify.a)


## liblibjpeg-turbo
add_library( liblibjpeg-turbo STATIC IMPORTED )
set_target_properties(liblibjpeg-turbo
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/liblibjpeg-turbo.a)


## liblibopenjp2
add_library( liblibopenjp2 STATIC IMPORTED )
set_target_properties(liblibopenjp2
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/liblibopenjp2.a)


## liblibpng
add_library( liblibpng STATIC IMPORTED )
set_target_properties(liblibpng
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/liblibpng.a)


## liblibprotobuf
add_library( liblibprotobuf STATIC IMPORTED )
set_target_properties(liblibprotobuf
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/liblibprotobuf.a)


## liblibtiff
add_library( liblibtiff STATIC IMPORTED )
set_target_properties(liblibtiff
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/liblibtiff.a)


## liblibwebp
add_library( liblibwebp STATIC IMPORTED )
set_target_properties(liblibwebp
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/liblibwebp.a)


## libquirc
add_library( libquirc STATIC IMPORTED )
set_target_properties(libquirc
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/libquirc.a)


## libtegra_hal
add_library( libtegra_hal STATIC IMPORTED )
set_target_properties(libtegra_hal
        PROPERTIES IMPORTED_LOCATION
        ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/opencv4/3rdparty/libtegra_hal.a)

## libvideo
add_library( libvideo STATIC IMPORTED)
set_target_properties(libvideo
        PROPERTIES IMPORTED_LOCATION
        PROPERTIES IMPORTED_LOCATION ${NATIVE_ROOT_PATH}/third_party/opencv/${OHOS_ARCH}/libopencv_video.a)

# OpenCV SHARED
set(OPENCV_LIB_PATH ${NATIVE_ROOT_PATH}/../../../libs/${OHOS_ARCH})




set(FSRMODELZOO mobilenetV2 ssd300 fpr depthanything rknnbenchmark)
foreach(EXE ${FSRMODELZOO})

    target_link_libraries(${EXE} PUBLIC libopencv_core )
    target_link_libraries(${EXE} PUBLIC libopencv_imgcodecs)
    target_link_libraries(${EXE} PUBLIC libopencv_imgproc)
    target_link_libraries(${EXE} PUBLIC libopencv_highgui)
    target_link_libraries(${EXE} PUBLIC libopencv_objdetect)
    target_link_libraries(${EXE} PUBLIC libopencv_calib3d)
    target_link_libraries(${EXE} PUBLIC libopencv_flann)
    target_link_libraries(${EXE} PUBLIC libopencv_features2d)
    target_link_libraries(${EXE} PUBLIC libopencv_dnn)
    target_link_libraries(${EXE} PUBLIC libopencv_videoio)
    target_link_libraries(${EXE} PUBLIC libvideo)
    target_link_libraries(${EXE} PUBLIC libade)
    target_link_libraries(${EXE} PUBLIC libittnotify)
    target_link_libraries(${EXE} PUBLIC liblibjpeg-turbo)
    target_link_libraries(${EXE} PUBLIC liblibopenjp2)
    target_link_libraries(${EXE} PUBLIC liblibpng)
    target_link_libraries(${EXE} PUBLIC liblibprotobuf)
    target_link_libraries(${EXE} PUBLIC liblibtiff)
    target_link_libraries(${EXE} PUBLIC liblibwebp)
    target_link_libraries(${EXE} PUBLIC libquirc)
    target_link_libraries(${EXE} PUBLIC libtegra_hal)
    target_link_libraries(${EXE} PUBLIC hilog_ndk.z)
    target_link_libraries(${EXE} PUBLIC ace_ndk.z)
    target_link_libraries(${EXE} PUBLIC ace_napi.z)
    target_link_libraries(${EXE} PUBLIC libc++.a)
    target_link_libraries(${EXE} PUBLIC z)
    target_link_libraries(${EXE} PUBLIC uv)
    target_link_libraries(${EXE} PUBLIC mindspore_lite_ndk)
    target_link_libraries(${EXE} PUBLIC rawfile.z libace_napi.z.so )


endforeach()

