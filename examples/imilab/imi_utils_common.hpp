// ============================================================
//              Imilab Utils: Common Definition
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_UTILS_COMMON_HPP__
#define __IMI_UTILS_COMMON_HPP__

#ifdef USE_OPENCV
/* opencv includes */
#include <opencv2/core/types.hpp>
#else // !USE_OPENCV
#include "imi_utils_types.hpp"
//#include "imi_imread.h"
#endif // USE_OPENCV

#ifdef USE_OPENCV
using namespace cv;
#endif // USE_OPENCV

typedef struct object_s {
    Rect2f rect;
    int label;
    float prob;
} Object;

typedef struct face_s : public Object {
    Point2f landmark[5];
} Face2f;

#endif // !__IMI_UTILS_COMMON_HPP__
