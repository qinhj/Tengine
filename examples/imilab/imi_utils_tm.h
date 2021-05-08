// @Author: qinhongjie@imilab.com

/* std c includes */
#include <stdio.h>  // for: printf
#include <float.h>  // for: DBL_MAX
/* tengine includes */
#include "common.h" // for: get_current_time
#include "tengine/c_api.h"      // for: graph_t, run_graph
#include "tengine_operations.h" // for: image

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
    fprintf(stdout, "[%s] Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
        __FUNCTION__, rc, total_time / rc, max_time, min_time);
    fprintf(stdout, "--------------------------------------\n");

    return 0;
}
