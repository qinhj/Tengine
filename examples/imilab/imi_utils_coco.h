// ============================================================
//                  Imilab Utils: COCO APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/10
// ============================================================

#ifndef __IMI_UTILS_COCO_H__
#define __IMI_UTILS_COCO_H__

static const int coco_class_num = 80;
static const char *coco_class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",            // 00-04
        "train", "truck", "boat", "traffic light",                              // 05-09
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",          // 10-14
        "cat", "dog", "horse", "sheep", "cow",                                  // 15-19
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",         // 20-24
        "handbag", "tie", "suitcase", "frisbee",                                // 25-29
        "skis", "snowboard", "sports ball", "kite", "baseball bat",             // 30-34
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", // 35-39
        "wine glass", "cup", "fork", "knife", "spoon",                          // 40-44
        "bowl", "banana", "apple", "sandwich", "orange",                        // 45-49
        "broccoli", "carrot", "hot dog", "pizza", "donut",                      // 50-54
        "cake", "chair", "couch", "potted plant", "bed",                        // 55-59
        "dining table", "toilet", "tv", "laptop", "mouse",                      // 60-64
        "remote", "keyboard", "cell phone", "microwave", "oven",                // 65-69
        "toaster", "sink", "refrigerator", "book", "clock",                     // 70-74
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"            // 75-79
};

//static const float coco_image_mean[] = { 0, 0, 0 };
//static const float coco_image_scale[] = { 0.003921, 0.003921, 0.003921 };

#endif // !__IMI_UTILS_COCO_H__
