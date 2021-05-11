// ============================================================
//                  Imilab Utils: Sort APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/10
// ============================================================

#ifndef __IMI_UTILS_SORT_H__
#define __IMI_UTILS_SORT_H__

/* std c++ includes */
#include <vector>
/* imilab includes */
#include "imi_utils_elog.h"

/** @brief
    @note:  _Tp must have member "rect"
*/
template<typename _Tp>
void nms_sorted_bboxes(const std::vector<_Tp>& objects, std::vector<int>& picked, float nms_threshold) {
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


/** @brief
    @note:  _Tp must have member "prob"
*/
template<typename _Tp>
void qsort_descent_inplace(std::vector<_Tp>& objects, int left, int right) {
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

/** @brief
    @note:  _Tp must have member "prob"
*/
template<typename _Tp>
void qsort_descent_inplace(std::vector<_Tp>& objects) {
    if (objects.empty()) return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

#endif // !__IMI_UTILS_SORT_H__
