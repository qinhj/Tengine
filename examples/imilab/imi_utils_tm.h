// ============================================================
//                  Imilab Utils: Tengine APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/08
// ============================================================

#ifndef __IMI_UTILS_TM_H__
#define __IMI_UTILS_TM_H__

/* std c includes */
#include <stdio.h>  // for: printf
#include <float.h>  // for: DBL_MAX
/* tengine includes */
#include "common.h" // for: get_current_time
#include "tengine/c_api.h"      // for: graph_t, run_graph
#include "tengine_operations.h" // for: image

typedef struct tm_tensor_output_s *tm_tensor_output_t;
typedef struct tm_tensor_output_s {
    tensor_t p_tensor;
    void *p_data;
    /* dequant parameter */
    float p_scale;
    int p_zero_point;
    int p_count;
} tm_tensor_output;

// @brief:  get output tensor info from input graph
// @param:  graph[in] graph
// @param:  pt[out] output tensor info
// @param:  cnt[in] output tensor count
// @param:  quant[in] uint8 quant flag
static int imi_utils_tm_get_graph_tensor(const graph_t &graph, tm_tensor_output_t pt, int cnt, char quant) {
    if (NULL == graph || cnt < 1 || NULL == pt) {
        return -1;
    }

    // ==================================================================
    // ========== This part is to get tensor information ================
    // ==================================================================
    for (int i = 0; i < cnt; i++) {
        pt[i].p_tensor = get_graph_output_tensor(graph, i, 0);
        pt[i].p_data = pt[i].p_tensor ? get_tensor_buffer(pt[i].p_tensor) : NULL;
        if (NULL == pt[i].p_data) {
            fprintf(stderr, "[%s] get_tensor_buffer NULL\n", __FUNCTION__);
            return -1;
        }
        //fprintf(stdout, "node[%d] p_tensor: %p, p_data: %p\n", i, pt[i].p_tensor, pt[i].p_data);
        if (quant) {
            /* dequant output data */
            get_tensor_quant_param(pt[i].p_tensor, &pt[i].p_scale, &pt[i].p_zero_point, 1);
            pt[i].p_count = get_tensor_buffer_size(pt[i].p_tensor) / sizeof(uint8_t);
            //fprintf(stdout, "p_count: %d, p_scale: %.4f, p_zero_point: %d\n",
            //    pt[i].p_count, pt[i].p_scale, pt[i].p_zero_point);
        }
    }
    return 0;
}

// @brief:  run tengine graph
// @param:  graph[in/out]
// @param:  rc[in] repeat count
// @return: error code
static int imi_utils_tm_run_graph(graph_t &graph, int rc) {
    double min_time = DBL_MAX, max_time = DBL_MIN, total_time = 0.;
    for (int i = 0; i < rc; i++) {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0) {
            fprintf(stderr, "[%s] Run graph failed\n", __FUNCTION__);
            return -1;
        }
        double end = get_current_time(), cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stdout, "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
        rc, total_time / rc, max_time, min_time);
    fprintf(stdout, "--------------------------------------\n");

    return 0;
}

#endif // !__IMI_UTILS_TM_H__
