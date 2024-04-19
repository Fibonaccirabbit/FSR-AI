//
// Created on 2024/4/16.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef ai_ssd_util_H
#define ai_ssd_util_H
#include <string>
#include <vector>
#include "Log.h"
#include "data_type.h"
class SSDModelUtil {
  public:
    // Constructor.
    SSDModelUtil(int srcImageWidth, int srcImgHeight);

    ~SSDModelUtil();

    /**
     * Return the SSD model post-processing result.
     * @param branchScores
     * @param branchBoxData
     * @return
     */
    void getDecodeResult(const float *& branchScores, const float *&, SSDInferResult &result);

    struct NormalBox {
        float y;
        float x;
        float h;
        float w;
    };

    struct YXBoxes {
        float ymin;
        float xmin;
        float ymax;
        float xmax;
    };

    struct Product {
        int x;
        int y;
    };

    struct WHBox {
        float boxw;
        float boxh;
    };

  private:
    std::vector<struct NormalBox> mDefaultBoxes;
    int inputImageHeight;
    int inputImageWidth;

    void getDefaultBoxes();

    void ssd_boxes_decode(const NormalBox *boxes,
                          YXBoxes *const decoded_boxes,
                          const float scale0 = 0.1, const float scale1 = 0.2,
                          const int count = 1917);

    void nonMaximumSuppression(const YXBoxes *const decoded_boxes, const float *const scores,
                               const std::vector<int> &in_indexes, std::vector<int> *out_indexes_p,
                               const float nmsThreshold = 0.6,
                               const int count = 1917, const int max_results = 1);

    double IOU(float r1[4], float r2[4]);

    // ============= variables =============.
    struct network {
        int model_input_height = 300;
        int model_input_width = 300;

        int num_default[6] = {3, 6, 6, 6, 6, 6};
        int feature_size[6] = {19, 10, 5, 3, 2, 1};
        double min_scale = 0.2;
        float max_scale = 0.95;
        float steps[6] = {16, 32, 64, 100, 150, 300};
        float prior_scaling[2] = {0.1, 0.2};
        float gamma = 2.0;
        float alpha = 0.75;
        int aspect_ratios[6][2] = {{0, 0},
                                   {2, 3},
                                   {2, 3},
                                   {2, 3},
                                   {2, 3},
                                   {2, 3}};
    } config;

    float g_thres_map[81] = {
        0,
        0.635,
        0.627,
        0.589,
        0.585,
        0.648,
        0.664,
        0.655,
        0.481,
        0.529,
        0.611,
        0.641,
        0.774,
        0.549,
        0.513,
        0.652,
        0.552,
        0.590,
        0.650,
        0.575,
        0.583,
        0.650,
        0.656,
        0.696,
        0.653,
        0.438,
        0.515,
        0.459,
        0.561,
        0.545,
        0.635,
        0.540,
        0.560,
        0.721,
        0.544,
        0.548,
        0.511,
        0.611,
        0.592,
        0.542,
        0.512,
        0.635,
        0.531,
        0.437,
        0.525,
        0.445,
        0.484,
        0.546,
        0.490,
        0.581,
        0.566,
        0.516,
        0.445,
        0.541,
        0.613,
        0.560,
        0.483,
        0.509,
        0.464,
        0.543,
        0.538,
        0.490,
        0.576,
        0.617,
        0.577,
        0.595,
        0.640,
        0.585,
        0.598,
        0.592,
        0.514,
        0.397,
        0.592,
        0.504,
        0.548,
        0.642,
        0.581,
        0.497,
        0.545,
        0.154,
        0.580,
    };

    std::string label_classes[81] = {
        {"background"},
        {"human"},
        {"bike"},
        {"automobile"},
        {"motorbike"},
        {"aircraft"},
        {"motorbus"},
        {"train"},
        {"motortruck"},
        {"boat"},
        {"traffic signal"},
        {"fireplug"},
        {"stop sign"},
        {"parking meter"},
        {"seat"},
        {"bird"},
        {"cat"},
        {"dog"},
        {"horse"},
        {"sheep"},
        {"cow"},
        {"elephant"},
        {"bear"},
        {"zebra"},
        {"giraffe"},
        {"knapsack"},
        {"bumbershoot"},
        {"purse"},
        {"neckwear"},
        {"traveling bag"},
        {"frisbee"},
        {"skis"},
        {"snowboard"},
        {"sports ball"},
        {"kite"},
        {"baseball bat"},
        {"baseball glove"},
        {"skateboard"},
        {"surfboard"},
        {"tennis racket"},
        {"bottle"},
        {"wine glass"},
        {"cup"},
        {"fork"},
        {"knife"},
        {"spoon"},
        {"bowl"},
        {"banana"},
        {"apple"},
        {"sandwich"},
        {"orange"},
        {"broccoli"},
        {"carrot"},
        {"hot dog"},
        {"pizza"},
        {"donut"},
        {"cake"},
        {"chair"},
        {"couch"},
        {"houseplant"},
        {"bed"},
        {"dinner table"},
        {"toilet"},
        {"television"},
        {"notebook computer"},
        {"mouse"},
        {"remote"},
        {"keyboard"},
        {"smartphone"},
        {"microwave"},
        {"oven"},
        {"toaster"},
        {"water sink"},
        {"fridge"},
        {"book"},
        {"bell"},
        {"vase"},
        {"shears"},
        {"toy bear"},
        {"hair drier"},
        {"toothbrush"}};
};

#endif // ai_ssd_util_H
