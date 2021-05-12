// ============================================================
//                  Imilab Utils: Yolo APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_UTILS_YOLOV5_HPP__
#define __IMI_UTILS_YOLOV5_HPP__

#define NODE_CNT_YOLOV5S    3

/* std c++ includes */
#include <cmath>    // for: round
/* imilab includes */
#include "imi_utils_imread.h"

/* focus process: 3x640x640 -> 12x320x320 */
/*
 | 0 2 |          C0-0, C1-0, C2-0,
 | 1 3 | x C3 =>  C0-1, C1-1, C2-1, x C12
                  C0-2, C1-2, C2-2,
                  C0-3, C1-3, C2-3,
*/
static int imi_utils_yolov5_focus_data(const float *data, image &lb) {
    for (int i = 0; i < 2; i++) {       // corresponding to rows
        for (int g = 0; g < 2; g++) {   // corresponding to cols
            for (int c = 0; c < lb.c; c++) {
                for (int h = 0; h < lb.h / 2; h++) {
                    for (int w = 0; w < lb.w / 2; w++) {
                        int in_index =
                            i + g * lb.w + c * lb.w * lb.h +
                            h * 2 * lb.w + w * 2;
                        int out_index =
                            i * 2 * lb.c * (lb.w / 2) * (lb.h / 2) +
                            g * lb.c * (lb.w / 2) * (lb.h / 2) +
                            c * (lb.w / 2) * (lb.h / 2) +
                            h * (lb.w / 2) +
                            w;

                        lb.data[out_index] = data[in_index];
                    }
                }
            }
        }
    }
    return 0;
}
static int imi_utils_yolov5_focus_data(const float *data, image &lb, float input_scale, int zero_point) {
    uint8_t *input_data = (uint8_t *)(lb.data);
    for (int i = 0; i < 2; i++) {       // corresponding to rows
        for (int g = 0; g < 2; g++) {   // corresponding to cols
            for (int c = 0; c < lb.c; c++) {
                for (int h = 0; h < lb.h / 2; h++) {
                    for (int w = 0; w < lb.w / 2; w++) {
                        int in_index =
                            i + g * lb.w + c * lb.w * lb.h +
                            h * 2 * lb.w + w * 2;
                        int out_index =
                            i * 2 * lb.c * (lb.w / 2) * (lb.h / 2) +
                            g * lb.c * (lb.w / 2) * (lb.h / 2) +
                            c * (lb.w / 2) * (lb.h / 2) +
                            h * (lb.w / 2) +
                            w;

                        /* quant to uint8 */
                        int udata = (int)round(data[in_index] / input_scale + (float)zero_point);
                        if (255 < udata) udata = 255;
                        else if (udata < 0) udata = 0;
                        input_data[out_index] = udata;
                    }
                }
            }
        }
    }
    return 0;
}

static int imi_utils_yolov5_load_data(FILE *fp, image &img, char bgr, image &lb, const float cov[][3], float input_scale, int zero_point) {
    // check inputs
    if (640 != lb.w || 640 != lb.h) {
        fprintf(stderr, "[%s] yolov5 letter box size must be: 640x640!\n", __FUNCTION__);
        exit(0);
    }
    // todo: optimize/resize input image
    if ((img.w <= img.h && 640 != img.h) ||
        (img.h <= img.w && 640 != img.w)) {
        fprintf(stderr, "[%s] input size (%d, %d) not match letter box size (%d, %d)!\n", __FUNCTION__, img.w, img.h, lb.w, lb.h);
        fprintf(stderr, "[%s] please try to resize the input image first!\n", __FUNCTION__);
        exit(0);
    }

    static float *data = (float *)calloc(sizeof(float), 640 * 640 * 3);
    //printf("mean:  %.3f, %.3f, %.3f\n", cov[0][0], cov[0][1], cov[0][2]);
    //printf("scale: %.3f, %.3f, %.3f\n", cov[1][0], cov[1][1], cov[1][2]);

    int img_size = img.w * img.h * img.c;
    if (NULL == img.data) {
        img.data = (float *)calloc(sizeof(float), img_size);
    }
    int lb_size = lb.w * lb.h * lb.c;
    if (NULL == lb.data) {
        lb.data = (float *)calloc(sizeof(float), lb_size);
    }

    float *swap = lb.data;
    lb.data = data;
    // load image to letter box
    int rc = imi_utils_load_letterbox(fp, img, bgr, lb, cov);
    if (1 != rc) {
        free(data);
        return rc;
    }
    lb.data = swap;

    // todo: optimize
    if (input_scale < 0) {
        return (imi_utils_yolov5_focus_data(data, lb), rc);
    }
    else {
        return (imi_utils_yolov5_focus_data(data, lb, input_scale, zero_point), rc);
    }
}

/* std c includes */
#include <float.h>  // for: FLT_MAX
/* std c++ includes */
#include <cmath>    // for: exp
/* imilab includes */
#include "imi_utils_coco.h"
#include "imi_utils_object.hpp"

static __inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

// todo: optimize
// @param:  feat[in] anchor results: box.x, box.y, box.w, box.h, box.score, {cls.score} x cls.num
static void imi_utils_yolov5_proposals_generate(
    int stride, const Size2i &lb, int anchor_group, const float *feat,
    float prob_threshold, std::vector<Object> &objects, int class_num = coco_class_num) {
    int anchor_num = 3;
    static float anchors[18] = { 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 };

    int cls_num = class_num; // 80
    int out_num = 4 + 1 + cls_num;// rent, score, cls_num

    int feat_w = lb.width / stride;
    int feat_h = lb.height / stride;
    for (int h = 0; h < feat_h; h++) {
        for (int w = 0; w < feat_w; w++) {
            for (int a = 0; a < anchor_num; a++) {
                int loc_idx = a * feat_w * feat_h * out_num + h * feat_w * out_num + w * out_num;
                // process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++) {
                    float score = feat[loc_idx + 5 + s];
                    if (score > class_score) {
                        class_index = s;
                        class_score = score;
                    }
                }
                // process box score
                float box_score = feat[loc_idx + 4];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold) {
                    float dx = sigmoid(feat[loc_idx + 0]);
                    float dy = sigmoid(feat[loc_idx + 1]);
                    float dw = sigmoid(feat[loc_idx + 2]);
                    float dh = sigmoid(feat[loc_idx + 3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    /*if (0 == class_index)*/
                    {
                        objects.push_back(obj);
                        //log_debug("[%s] stride: %d, push back: %d\n", __FUNCTION__, stride, class_index);
                    }
                }
            }
        }
    }
}

#endif // !__IMI_UTILS_YOLOV5_HPP__
