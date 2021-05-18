// ============================================================
//              Imilab Utils: Tengine Quant APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/08
// ============================================================

#ifndef __IMI_UTILS_TM_QUANT_H__
#define __IMI_UTILS_TM_QUANT_H__

/* std c includes */
#include <stdio.h>  // for: fprintf
/* tengine includes */
#include "tengine/c_api.h"  // for: tensor_t, get_tensor_buffer
/* imilab includes */
#include "imi_utils_tm-inl.h"

typedef struct tm_quant_s *tm_quant_t;
typedef struct tm_quant_s {
    /* buffer for outputs */
    const void *buffer;
    int size;
    /* dequant parameter */
    float scale;
    int zero_point;
    /* user data */
    void *_data;
} tm_quant;

static void __inline imi_utils_tm_quant_free(tm_quant_t quant) {
    if (quant) {
        if (quant->_data) free(quant->_data);
        free(quant);
    }
}

// @brief:  get quant info from input tensor
// @param:  tensor[in]
// @param:  quant[out]
static int imi_utils_tm_quant_get(tensor_t tensor, tm_quant_t quant) {
    if (NULL == tensor || NULL == quant) {
        return -1;
    }

    size_t dt = get_tensor_data_type(tensor);
    // ===========================================================
    // ========== This part is to get quant information ==========
    // ===========================================================
    quant->buffer = get_tensor_buffer(tensor);
    if (NULL == quant->buffer) {
        fprintf(stderr, "[%s] get tensor[%p] buffer NULL\n", __FUNCTION__, tensor);
        return -1;
    }
    quant->size = get_tensor_buffer_size(tensor) / imi_utils_tm_get_tensor_datasize(dt);
    // todo: update to quant param array
    get_tensor_quant_param(tensor, &quant->scale, &quant->zero_point, 1);
    fprintf(stdout, "tensor[%p] data: %p, size: %d, scale: %.4f, zero_point: %d\n",
        tensor, quant->buffer, quant->size, quant->scale, quant->zero_point);
    return 0;
}

#endif // !__IMI_UTILS_TM_QUANT_H__
