// ============================================================
//              Imilab Utils: Proposal Object APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_UTILS_OBJECT_HPP__
#define __IMI_UTILS_OBJECT_HPP__

/* std c includes */
#include <stdio.h>  // for: stdout
/* std c++ includes */
#include <vector>
/* imilab includes */
#include "imi_utils_elog.h"
#include "imi_utils_common.hpp" // for: Object

#define DFLT_THRESHOLD_NMS  0.45f

// @brief:  qsort_descent_inplace
// @note:   _Tp must have member "prob"
template<typename _Tp>
static __inline void qsort_descent_inplace(std::vector<_Tp> &objects, int left, int right) {
    int i = left, j = right;
    float p = (float)objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);
            i++, j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

// @brief:  qsort_descent_inplace
// @note:   _Tp must have member "prob"
template<typename _Tp>
void imi_utils_objects_qsort(std::vector<_Tp> &objects, char descent = 1) {
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// @brief:  nms sorted bboxes
// @note:   _Tp must have member "rect"
template<typename _Tp>
void imi_utils_objects_nms(const std::vector<_Tp>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = objects.size();
    log_debug("[%s] proposal num: %d\n", __FUNCTION__, n);

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        const _Tp &obj = objects[i];
        areas[i] = obj.rect.area();
        log_debug("[%s] object[%d] class[%d] (%.3f, %.3f), w: %.3f, h: %.3f\n", __FUNCTION__,
            i, obj.label, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }

    for (int i = 0; i < n; i++) {
        const _Tp &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size() && keep; j++) {
            const _Tp &b = objects[picked[j]];

            // intersection over union
            float inter_area = (a.rect & b.rect).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area / union_area;
            log_debug("[%s] IOU(%d, %d)=%.3f/%.3f=%.3f\n", __FUNCTION__, i, j, inter_area, union_area, iou);
            // float IoU = inter_area / union_area
            if (iou > nms_threshold) {
                keep = 0;
            }
        }

        if (keep) {
            picked.push_back(i);
            log_debug("[%s] push face(%d) into picked list\n", __FUNCTION__, i);
        }
    }
}

// @brief:  filter proposal objects
template<typename _Tp>
std::vector<_Tp> imi_utils_proposals_filter(const std::vector<_Tp> &proposals,
    const Size2i &image_size, const Size2i &letter_box, float nms_threshold = DFLT_THRESHOLD_NMS) {
    std::vector<int> picked;
    // apply nms with nms_threshold
    imi_utils_objects_nms(proposals, picked, nms_threshold);

    int count = picked.size();
    log_debug("[%s] %d objects are picked\n", __FUNCTION__, count);

    // post process: scale and offset for letter box
    Size2i lb_offset;
    float lb_scale = -1;
    if (0 < letter_box.width && 0 < letter_box.height) {
        float scale_w = letter_box.width * 1.0 / image_size.width;
        float scale_h = letter_box.height * 1.0 / image_size.height;
        lb_scale = scale_h < scale_w ? scale_h : scale_w;
        lb_offset.width = int(lb_scale * image_size.width);
        lb_offset.height = int(lb_scale * image_size.height);
        lb_offset.width = (letter_box.width - lb_offset.width) / 2;
        lb_offset.height = (letter_box.height - lb_offset.height) / 2;
        log_debug("[%s] letter box scale: %3.4f, offset: (%d, %d)\n",
            __FUNCTION__, lb_scale, lb_offset.width, lb_offset.height);
    }

    std::vector<_Tp> objects(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        // post process: from letter box to input image
        if (0 < letter_box.width && 0 < letter_box.height) {
            x0 = (x0 - lb_offset.width) / lb_scale;
            y0 = (y0 - lb_offset.height) / lb_scale;
            x1 = (x1 - lb_offset.width) / lb_scale;
            y1 = (y1 - lb_offset.height) / lb_scale;
        }

        x0 = std::max(std::min(x0, (float)(image_size.width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(image_size.height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(image_size.width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(image_size.height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return objects;
}

#endif // !__IMI_UTILS_OBJECT_HPP__
