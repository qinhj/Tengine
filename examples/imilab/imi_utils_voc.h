// ============================================================
//                  Imilab Utils: VOC APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/25
// ============================================================

#ifndef __IMI_UTILS_VOC_H__
#define __IMI_UTILS_VOC_H__

static const int voc_class_num = 21;
static const char *voc_class_names[] = {
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",       // 01-05
    "bus", "car", "cat", "chair", "cow",                    // 06-10
    "diningtable", "dog", "horse", "motorbike", "person",   // 11-15
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",   // 16-20
};

//static const float voc_image_mean[] = { 127.5f, 127.5f, 127.5f };
//static const float voc_image_scale[] = { 0.007843f, 0.007843f, 0.007843f };
static const float voc_image_cov[][3] = {
    { 127.5f, 127.5f, 127.5f }, // mean
    { 0.007843f, 0.007843f, 0.007843f } // scale
};

#endif // !__IMI_UTILS_VOC_H__
