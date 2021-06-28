// ============================================================
//                  Imilab Model: Yolov5 APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_MODEL_YOLOV5_HPP__
#define __IMI_MODEL_YOLOV5_HPP__

/* imilab includes */
#include "imi_model_yolov3.hpp"

#define NODE_CNT_YOLOV5S        3
#define NODE_CNT_YOLOV5S_TINY   2

// yolov5s-tiny anchors
static const int anchors_tiny_v5s[] = {
     9,  17,  18,  39,  37, 67, // P3/8
    62, 132, 120, 242, 273, 363 // P4/16
};

// yolov5s input info
static yolov3 yolov5s = {
    make_empty_image(640, 640, 3),
    NODE_CNT_YOLOV5S,
    coco_class_num,
    coco_class_names,
    /* hyp */
    strides,
    3, anchors,
    coco_image_cov
};

// yolov5s-tiny input info
static yolov3 yolov5s_tiny = {
    make_empty_image(640, 640, 3),
    NODE_CNT_YOLOV5S_TINY,
    coco_class_num,
    coco_class_names,
    /* hyp */
    strides,
    3, anchors_tiny_v5s,
    coco_image_cov
};

/* std c++ includes */
#include <cmath>    // for: round
/* imilab includes */
#include "imi_utils_image.h"

/* focus process: 3x640x640 -> 12x320x320 */
/*
 | 0 2 |          C0-0, C1-0, C2-0,
 | 1 3 | x C3 =>  C0-1, C1-1, C2-1, x C12
                  C0-2, C1-2, C2-2,
                  C0-3, C1-3, C2-3,
*/
static int imi_utils_yolov5_focus_data(const float *data, image &lb) {
    // check inputs
    if (NULL == data || NULL == lb.data) {
        return -1;
    }

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
    // check inputs
    if (NULL == data || NULL == lb.data) {
        return -1;
    }

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

static int imi_utils_yolov5_load_data(FILE *fp, image &img, char bgr, image &lb, const float cov[2][3], float input_scale, int zero_point) {
    // check buffer data
    if (NULL == lb.data) {
        log_error("letter box data buffer is NULL\n");
        return -2;
    }

    static float *data = (float *)calloc(sizeof(float), lb.w * lb.h * lb.c);
    //log_echo("mean:  %.3f, %.3f, %.3f\n", cov[0][0], cov[0][1], cov[0][2]);
    //log_echo("scale: %.3f, %.3f, %.3f\n", cov[1][0], cov[1][1], cov[1][2]);

    if (NULL == img.data) {
        img.data = (float *)calloc(sizeof(float), img.w * img.h * img.c);
    }

    float *swap = lb.data;
    lb.data = data;
    // load image to letter box
    int rc = imi_utils_image_load_letterbox(fp, img, bgr, lb, cov);
    lb.data = swap;
    if (1 != rc) {
        free(data);
        return rc;
    }

    // todo: optimize (C, H, W) -> (C*4, H/2, W/2)
    if (input_scale < 0) {
        return (imi_utils_yolov5_focus_data(data, lb), rc);
    }
    else {
        return (imi_utils_yolov5_focus_data(data, lb, input_scale, zero_point), rc);
    }
}

#endif // !__IMI_MODEL_YOLOV5_HPP__
