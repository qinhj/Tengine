// ============================================================
//                  Imilab Utils: Tengine APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/08
// ============================================================

#ifndef __IMI_UTILS_TM_H__
#define __IMI_UTILS_TM_H__

/* std c includes */
#include <stdio.h>  // for: printf
#include <stdlib.h> // for: calloc
#include <float.h>  // for: DBL_MAX
/* tengine includes */
#include "common.h" // for: get_current_time
#include "tengine/c_api.h"      // for: graph_t, run_graph
#include "tengine_operations.h" // for: image

typedef struct tm_tensor_s tm_tensor_input_s;
typedef struct tm_tensor_s tm_tensor_output_s;
typedef struct tm_tensor_s *tm_tensor_input_t;
typedef struct tm_tensor_s *tm_tensor_output_t;
typedef struct tm_tensor_s *tm_tensor_t;
typedef struct tm_tensor_s {
    tensor_t p_tensor;
    void *p_data;
    int p_count;
    /* dequant parameter */
    float p_scale;
    int p_zero_point;
} tm_tensor;

// @brief:  get output tensor info from input graph
// @param:  graph[in] graph
// @param:  pt[out] output tensor info
// @param:  cnt[in] output tensor count
// @param:  quant[in] uint8 quant flag
static int imi_utils_tm_get_graph_tensor(const graph_t &graph, tm_tensor_t pt, int cnt, char quant) {
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
        pt[i].p_count = get_tensor_buffer_size(pt[i].p_tensor) / sizeof(float);
        if (quant) {
            /* dequant output data */
            get_tensor_quant_param(pt[i].p_tensor, &pt[i].p_scale, &pt[i].p_zero_point, 1);
            pt[i].p_count *= (sizeof(float) / sizeof(uint8_t));
            fprintf(stdout, "p_scale: %.4f, p_zero_point: %d\n", pt[i].p_scale, pt[i].p_zero_point);
        }
        fprintf(stdout, "node[%d] p_tensor: %p, p_data: %p, p_count: %d\n",
            i, pt[i].p_tensor, pt[i].p_data, pt[i].p_count);
    }
    return 0;
}

// @param:  check[in]   whether check the value of dim or not
static __inline int _imi_utils_tm_show_tensor(const tensor_t &tensor, int idx, char quant, int check) {
    fprintf(stdout, "  tensor[%2d] name: %s\n", idx, get_tensor_name(tensor));
    fprintf(stdout, "  tensor[%2d] type: %d\n", idx, get_tensor_data_type(tensor)); // 0: FP32; 3: UINT8
    fprintf(stdout, "  tensor[%2d] layout: %s\n", idx, get_tensor_layout(tensor) ? "nhwc" : "nchw");
    fprintf(stdout, "  tensor[%2d] buffer: %p, size: %d\n", idx, get_tensor_buffer(tensor),
        quant ? get_tensor_buffer_size(tensor) : get_tensor_buffer_size(tensor) / 4);
    if (quant) {
        static float scale;
        static int zero_point;
        get_tensor_quant_param(tensor, &scale, &zero_point, 1);
        fprintf(stdout, "  tensor[%2d] scale: %.4f, zero_point: %d\n", idx, scale, zero_point);
    }
    int dims[4], rc = 0;
    int dim_num = get_tensor_shape(tensor, dims, 4);
    fprintf(stdout, "  tensor[%2d] dim num: %d(", idx, dim_num);
    for (int i = 0; i < dim_num; i++) {
        if (check && dims[i] < 1) rc++;
        fprintf(stdout, " %d", dims[i]);
    }
    fprintf(stdout, ")\n");
    return rc;
}
static __inline int _imi_utils_tm_show_node(const node_t &node, char quant, int check) {
    tensor_t tensor;
    int count, idx, rc = 0;
    count = get_node_input_number(node);
    fprintf(stdout, "input tensor count: %d\n", count);
    for (idx = 0; idx < count; idx++) {
        tensor = get_node_input_tensor(node, idx);
        rc += _imi_utils_tm_show_tensor(tensor, idx, quant, check);
    }

    count = get_node_output_number(node);
    fprintf(stdout, "output tensor count: %d\n", count);
    for (idx = 0; idx < count; idx++) {
        tensor = get_node_output_tensor(node, idx);
        rc += _imi_utils_tm_show_tensor(tensor, idx, quant, check);
    }
    return rc;
}
// @brief:  show graph input/output tensor info
static int imi_utils_tm_show_graph(const graph_t &graph, char quant) {
    node_t node;
    int count, idx, rc = 0;

#if 0
    count = get_graph_input_node_number(graph);
    fprintf(stdout, "===========================================\n");
    fprintf(stdout, "------------- input node info -------------\n");
    fprintf(stdout, "===========================================\n");
    fprintf(stdout, "input node count: %d\n", count);
    for (idx = 0; idx < count; idx++) {
        node = get_graph_input_node(graph, idx);
        fprintf(stdout, "-------------------------------------------\n");
        fprintf(stdout, "node[%2d] name: %s\n", idx, get_node_name(node));
        fprintf(stdout, "node[%2d] op:   %s\n", idx, get_node_op(node));
        rc += _imi_utils_tm_show_node(node, quant, 0);
    }
#endif

#if 0
    count = get_graph_output_node_number(graph);
    fprintf(stdout, "===========================================\n");
    fprintf(stdout, "------------- output node info -------------\n");
    fprintf(stdout, "===========================================\n");
    fprintf(stdout, "output node count: %d\n", count);
    for (idx = 0; idx < count; idx++) {
        node = get_graph_output_node(graph, idx);
        fprintf(stdout, "-------------------------------------------\n");
        fprintf(stdout, "node[%2d] name: %s\n", idx, get_node_name(node));
        fprintf(stdout, "node[%2d] op:   %s\n", idx, get_node_op(node));
        rc += _imi_utils_tm_show_node(node, quant, 0);
    }
#endif

#if 1
    count = get_graph_node_num(graph);
    fprintf(stdout, "===========================================\n");
    fprintf(stdout, "------------- graph node info -------------\n");
    fprintf(stdout, "===========================================\n");
    fprintf(stdout, "graph node count: %d\n", count);
    for (idx = 0; idx < count; idx++) {
        node = get_graph_node_by_idx(graph, idx);
        fprintf(stdout, "-------------------------------------------\n");
        fprintf(stdout, "node[%2d] name: %s\n", idx, get_node_name(node));
        fprintf(stdout, "node[%2d] op:   %s\n", idx, get_node_op(node));
        rc += _imi_utils_tm_show_node(node, quant, 0 /*1*/);
    }
#endif

    fprintf(stdout, "\n");
    if (rc) {
        fprintf(stderr, "[Error] %d dims are <= 0\n", rc);
    }
    return rc;
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
        min_time = min_time < cur ? min_time : cur;
        max_time = max_time < cur ? cur : max_time;
    }
    fprintf(stdout, "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
        rc, total_time / rc, max_time, min_time);
    fprintf(stdout, "--------------------------------------\n");

    return 0;
}

#endif // !__IMI_UTILS_TM_H__
