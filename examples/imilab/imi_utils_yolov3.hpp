// ============================================================
//                  Imilab Utils: Yolov3 APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_UTILS_YOLOV3_HPP__
#define __IMI_UTILS_YOLOV3_HPP__

/* tengine includes */
#include "tengine_operations.h" // for: image

// output node count
#define NODE_CNT_YOLOV3         3
#define NODE_CNT_YOLOV3_TINY    2

typedef struct yolov3_s {
    int anchors_num;
    const float *anchors;
    image lb;       // letter box size
    //int node_cnt;   // output node num
} yolov3;

// yolov3 anchors
static const float anchors[] = {
    10, 13, 16, 30, 33, 23,         // P3/8
    30, 61, 62, 45, 59, 119,        // P4/16
    116, 90, 156, 198, 373, 326,    // P5/32
};
// yolov3-tiny anchors
static const float anchors_tiny[] = {
    10, 14, 23, 27, 37, 58,         // P4/16
     81, 82, 135, 169, 344, 319,    // P5/32
};

// yolov3 input info
static yolov3 yolov3_std = {
    9, anchors, make_empty_image(608, 608, 3)
};
// yolov3-tiny input info
static yolov3 yolov3_tiny = {
    6, anchors_tiny, make_empty_image(416, 416, 3)
};

#endif // !__IMI_UTILS_YOLOV3_HPP__
