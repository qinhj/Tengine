// ============================================================
//              Imilab Utils: Tengine Inline APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/17
// ============================================================

#ifndef __IMI_UTILS_TM_INLINE_H__
#define __IMI_UTILS_TM_INLINE_H__

/* std c includes */
#include <stdio.h> // for: size_t
/* tengine includes */
#include "tengine/c_api.h"

#ifndef TENGINE_DT_MAX
#define TENGINE_DT_MAX 6
#endif // !TENGINE_DT_MAX

// @brief:  get tensor datatype name
static __inline const char* imi_utils_tm_get_tensor_datatype(size_t type)
{
    static const char* _tensor_dt_name[] = {
        "TENGINE_DT_FP32", // 0
        "TENGINE_DT_FP16",
        "TENGINE_DT_INT8",
        "TENGINE_DT_UINT8", // 3
        "TENGINE_DT_INT32",
        "TENGINE_DT_INT16",
        "TENGINE_DT_UNKNOWN"};
    return type < TENGINE_DT_MAX ? _tensor_dt_name[type] : _tensor_dt_name[TENGINE_DT_MAX];
}

// @brief:  get tensor datatype size
static __inline int imi_utils_tm_get_tensor_datasize(size_t type)
{
    switch (type)
    {
    case TENGINE_DT_FP32:
    case TENGINE_DT_INT32:
        return 4;
    case TENGINE_DT_FP16:
    case TENGINE_DT_INT16:
        return 2;
    default:
        return 1;
    }
    return -1;
}

#endif // !__IMI_UTILS_TM_INLINE_H__
