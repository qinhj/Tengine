// ============================================================
//                  Imilab Utils: Sort APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/10
// ============================================================

#ifndef __IMI_UTILS_SORT_H__
#define __IMI_UTILS_SORT_H__

/* std c++ includes */
#include <vector>

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
