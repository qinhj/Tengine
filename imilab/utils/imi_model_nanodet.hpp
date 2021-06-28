// ============================================================
//                  Imilab Model: NanoDet APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/06/28
// ============================================================

#ifndef __IMI_MODEL_NANODET_HPP__
#define __IMI_MODEL_NANODET_HPP__

/* tengine includes */
#include "tengine_operations.h" // for: image
/* imilab includes */
#include "imi_utils_elog.h"     // for: log_xxxx
#include "imi_utils_coco.h"     // for: coco_class_names
#include "imi_utils_object.hpp"

// max output branch count
#define BR_CNT_NANODET  5

// nanodet strides
static const int strides[] = { 8, 16, 32 };
// nanodet strides tiny
static const int strides_tiny[] = { 16, 32 };

// nanodet output tensor names
const char* cls_pred_name[] = {
    "cls_pred_stride_8", "cls_pred_stride_16", "cls_pred_stride_32"
};
const char* dis_pred_name[] = {
    "dis_pred_stride_8", "dis_pred_stride_16", "dis_pred_stride_32"
};

// nanodet image mean && norm
static const float nano_image_cov[][3] = {
    { 103.53f, 116.28f, 123.675f }, // bgr
    { 0.017429f, 0.017507f, 0.017125f },
};

typedef struct nanodet_s {
    /* input */
    image lb;       // letter box size as model input shape
    /* output */
    int outputs;    // output branch count
    const char* const *cls_pred_name;
    const char* const *dis_pred_name;
    /* label info */
    int class_num;
    const char* const *class_names;
    /* hyp */
    const int reg_max;
    const int *strides;
    const void *usr_data;
} nanodet;

// nanodet model: m
static nanodet nanodet_m = {
    make_empty_image(320, 320, 3),
    3, cls_pred_name, dis_pred_name, 
    coco_class_num, coco_class_names,
    8, strides, nano_image_cov,
};

template<typename _Tp>
static int softmax(const _Tp* src, _Tp* dst, int length) {
    const _Tp max_value = *std::max_element(src, src + length);
    _Tp denominator{ 0 };
 
    for (int i = 0; i < length; ++i) {
        dst[i] = std::exp(src[i] - max_value);
        denominator += dst[i];
    }
 
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

// @brief:  generate and filter proposals
// @param:  model[in]   nanodet model info
// @param:  data[in]    output tensor buffer()
//  cls_pred (1, num_grid, cls_num), dis_pred (1, num_grid, 4*reg_max)
// @param:  threshold[in]
// @param:  objects[out] output detected objects
static int imi_utils_nanodet_proposals_generate(const nanodet &model,
    const void *data[], float threshold, std::vector<Object>& objects) {
    // Discrete distribution parameter, see the following resources for more details:
    // [nanodet-m.yml](https://github.com/RangiLyu/nanodet/blob/main/config/nanodet-m.yml)
    // [GFL](https://arxiv.org/pdf/2006.04388.pdf)
    const int reg_max_1 = model.reg_max;  // 32 / 4;
    const int class_num = model.class_num;
    const image &in_pad = model.lb;

    for (int n = 0; n < model.outputs; n++) {
        const int stride = model.strides[n];
        const int num_grid_x = in_pad.w / stride;
        const int num_grid_y = in_pad.h / stride;
        const float *cls_pred = (const float *)data[2 * n + 0];
        const float *dis_pred = (const float *)data[2 * n + 1];

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const int idx = i * num_grid_x + j;
                const float *scores = cls_pred + idx * class_num;

                // find label with max score
                int label = -1;
                float score = -FLT_MAX;
                for (int k = 0; k < class_num; k++) {
                    if (score < scores[k]) {
                        label = k, score = scores[k];
                    }
                }

                if (threshold < score) {
                    float pred_ltrb[4];
                    for (int k = 0; k < 4; k++) {
                        float dis = 0.f;
                        // predicted distance distribution after softmax
                        float dis_after_sm[8] = { 0. };
                        softmax(dis_pred + idx * reg_max_1 * 4 + k * reg_max_1, dis_after_sm, 8);
                        // integral on predicted discrete distribution
                        for (int l = 0; l < reg_max_1; l++) {
                            dis += l * dis_after_sm[l];
                            //log_debug("%2.6f ", dis_after_sm[l]);
                        }
                        //log_debug("\n");

                        pred_ltrb[k] = dis * stride;
                    }

                    // predict box center point
                    float pb_cx = (j + 0.5f) * stride;
                    float pb_cy = (i + 0.5f) * stride;

                    float x0 = pb_cx - pred_ltrb[0]; // left
                    float y0 = pb_cy - pred_ltrb[1]; // top
                    float x1 = pb_cx + pred_ltrb[2]; // right
                    float y1 = pb_cy + pred_ltrb[3]; // bottom

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = label;
                    obj.prob = score;

                    objects.push_back(obj);
                }
            }
        }
    }

    return objects.size();
}

#endif // !__IMI_MODEL_NANODET_HPP__
