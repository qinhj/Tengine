// ============================================================
//                  Imilab Utils: Tengine APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/08
// ============================================================

#ifndef __IMI_UTILS_TM_H__
#define __IMI_UTILS_TM_H__

/* std c includes */
#include <stdio.h>  // for: fprintf
#include <float.h>  // for: DBL_MAX
/* tengine includes */
#include "common.h" // for: get_current_time
#include "tengine/c_api.h"  // for: graph_t, run_graph

static int total_cnt = 0;
static double min_time = DBL_MAX, max_time = DBL_MIN, total_time = 0.;

// @brief:  get tengine runtime status
static __inline void imi_utils_tm_run_status(void *data) {
    fprintf(stdout, "======================================\n");
    fprintf(stdout, "Run graph %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
        total_cnt, total_time / total_cnt, max_time, min_time);
    fprintf(stdout, "======================================\n");
}

// @brief:  run tengine graph
// @param:  graph[in/out]
// @param:  rc[in] repeat count
// @return: error code
static int imi_utils_tm_run_graph(graph_t graph, int rc) {
    int i;
    double start, cur;
    for (i = 0; i < rc; i++) {
        start = get_current_time();
        if (run_graph(graph, 1) < 0) {
            fprintf(stderr, "[%s] Run graph failed\n", __FUNCTION__);
            return -1;
        }
        cur = get_current_time() - start;
        total_time += cur;
        min_time = min_time < cur ? min_time : cur;
        max_time = max_time < cur ? cur : max_time;
    }
    total_cnt += i;
    return 0;
}

#endif // !__IMI_UTILS_TM_H__
