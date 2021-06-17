// ============================================================
//                  Imilab Utils: COCO APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/10
// ============================================================

#ifndef __IMI_UTILS_COCO_H__
#define __IMI_UTILS_COCO_H__

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
static const int coco_class_num = sizeof(coco_class_names) / sizeof(coco_class_names[0]);

//static const float coco_image_mean[] = { 0, 0, 0 };
//static const float coco_image_scale[] = { 0.003921, 0.003921, 0.003921 };
static const float coco_image_cov[][3] = {
    { 0, 0, 0 }, // mean
    { 0.003921, 0.003921, 0.003921 } // scale
};


#include "imi_utils_elog.h" // for: log_xxxx

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// @brief:  show usage
static __inline void show_usage(const char *exe, const char *model[2]) {
    const char *tests[] = {
        "imilab_640x360x3_bgr_catdog.rgb24",
        "imilab_640x360x3_bgr_human1.rgb24",
        "imilab_640x360x3_bgr_human2.rgb24",
        "imilab_960x512x3_bgr_human3.rgb24",
    };
    log_echo("[Usage]:  [-u]\n");
    log_echo("    [-m model_file] [-i input_file] [-o output_file] [-n class_number]\n");
    log_echo("    [-w width] [-h height] [-c target_class] [-s threshold] [-f max_frame]\n");
    log_echo("    [-r repeat_count] [-t thread_count]\n");
    log_echo("[Examples]:\n");
    log_echo("   # coco 80 classes\n");
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 200\n", exe, model[0], tests[0], tests[0]);
    log_echo("   # specific class of coco 80 classes(e.g. person)\n");
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 100 -c 0\n", exe, model[0], tests[1], tests[1]);
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 500 -c 0\n", exe, model[0], tests[2], tests[2]);
    log_echo("   # single class(e.g. person)\n");
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 200 -n 1\n", exe, model[1], tests[0], tests[0]);
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 100 -n 1\n", exe, model[1], tests[1], tests[1]);
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 500 -n 1\n", exe, model[1], tests[2], tests[2]);
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 500 -n 1 -w 960 -h 512\n", exe, model[1], tests[3], tests[3]);
}

static __inline int parse_args(void *data) {
    // todo: ...
    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !__IMI_UTILS_COCO_H__
