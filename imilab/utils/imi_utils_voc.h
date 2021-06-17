// ============================================================
//                  Imilab Utils: VOC APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/25
// ============================================================

#ifndef __IMI_UTILS_VOC_H__
#define __IMI_UTILS_VOC_H__

static const char *voc_class_names[] = {
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",       // 01-05
    "bus", "car", "cat", "chair", "cow",                    // 06-10
    "diningtable", "dog", "horse", "motorbike", "person",   // 11-15
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",   // 16-20
};
static const int voc_class_num = sizeof(voc_class_names) / sizeof(voc_class_names[0]);

//static const float voc_image_mean[] = { 127.5f, 127.5f, 127.5f };
//static const float voc_image_scale[] = { 0.007843f, 0.007843f, 0.007843f };
static const float voc_image_cov[][3] = {
    { 127.5f, 127.5f, 127.5f }, // mean
    { 0.007843f, 0.007843f, 0.007843f } // scale
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
    log_echo("   # voc 21 classes\n");
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 200\n", exe, model[0], tests[0], tests[0]);
    log_echo("   # specific class of voc 21 classes(e.g. '-c 15' as person)\n");
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 100 -c 15\n", exe, model[0], tests[1], tests[1]);
    log_echo("   %s -m %s -i %s -o output/%s -t 4 -f 500 -c 15\n", exe, model[0], tests[2], tests[2]);
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

#endif // !__IMI_UTILS_VOC_H__
