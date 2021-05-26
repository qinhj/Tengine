// ============================================================
//              Imilab Utils: Tengine Debug APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/17
// ============================================================

#ifndef __IMI_UTILS_TM_DEBUG_H__
#define __IMI_UTILS_TM_DEBUG_H__

/* std c includes */
#include <stdio.h>  // for: printf
/* tengine includes */
#include "tengine/c_api.h"  // for: graph_t, run_graph
/* imilab includes */
#include "imi_utils_tm-inl.h"

// @brief:  show tensor buffer data
// @param:  data[in]    tensor buffer pointer
// @param:  dt[in]      tensor data type
// @param:  size[in]    tensor data size
static __inline void _imi_utils_tm_show_tensor_buffer(const void *data, size_t dt, size_t size) {
    size_t i;
    if (TENGINE_DT_FP32 == dt) {
        const float *_data = (const float *)data;
        for (i = 0; i < size; i++) {
            fprintf(stdout, " %.3f", _data[i]);
        }
    }
    else if (TENGINE_DT_INT8 == dt) {
        const char *_data = (const char *)data;
        for (i = 0; i < size; i++) {
            fprintf(stdout, " %d", _data[i]);
        }
    }
    else if (TENGINE_DT_UINT8 == dt) {
        const unsigned char *_data = (const unsigned char *)data;
        for (i = 0; i < size; i++) {
            fprintf(stdout, " %u", _data[i]);
        }
    }
    else if (TENGINE_DT_INT32 == dt) {
        const int *_data = (const int *)data;
        for (i = 0; i < size; i++) {
            fprintf(stdout, " %d", _data[i]);
        }
    }
    else if (TENGINE_DT_INT16 == dt) {
        const short *_data = (const short *)data;
        for (i = 0; i < size; i++) {
            fprintf(stdout, " %hd", _data[i]);
        }
    }
    else {
        fprintf(stdout, "%s not support to show yet!\n", imi_utils_tm_get_tensor_datatype(dt));
    }

    fprintf(stdout, "\n");
    return;
}

// @brief:  show tensor info
// @param:  check[in]   whether check the value of dim or not
static __inline int _imi_utils_tm_show_tensor(const tensor_t tensor, int idx, char quant, int check) {
    size_t dt = get_tensor_data_type(tensor);
    void *buf = get_tensor_buffer(tensor);
    int buf_size = get_tensor_buffer_size(tensor) / imi_utils_tm_get_tensor_datasize(dt);
    fprintf(stdout, "  tensor[%2d] name: %s\n", idx, get_tensor_name(tensor));
    fprintf(stdout, "  tensor[%2d] type: %s\n", idx, imi_utils_tm_get_tensor_datatype(dt));
    fprintf(stdout, "  tensor[%2d] layout: %s\n", idx, get_tensor_layout(tensor) ? "nhwc" : "nchw");
    fprintf(stdout, "  tensor[%2d] buffer: %p, size: %d\n", idx, buf, buf_size);
    // todo: check and call tengine api to get more quant info
    if (quant) {
        // todo: update hard coded "1" as quant array length
        static float scale;
        static int zero_point;
        get_tensor_quant_param(tensor, &scale, &zero_point, 1);
        fprintf(stdout, "  tensor[%2d] scale: %.4f, zero_point: %d\n", idx, scale, zero_point);
    }
    // show output dims
    int dims[5], rc = 0;
    int dim_num = get_tensor_shape(tensor, dims, 5); // check at most 5 dims
    fprintf(stdout, "  tensor[%2d] dim num: %d(", idx, dim_num);
    for (int i = 0; i < dim_num; i++) {
        if (check && dims[i] < 1) rc++;
        fprintf(stdout, " %d", dims[i]);
    }
    fprintf(stdout, ")\n");
    // show tensor data if dim is 1
    if (1 == dim_num) {
        fprintf(stdout, "  tensor[%2d] data:", idx);
        _imi_utils_tm_show_tensor_buffer(get_tensor_buffer(tensor), dt, buf_size);
    }
    return rc;
}
// @brief:  show node info
static __inline int _imi_utils_tm_show_node(const node_t node, char quant, int check) {
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

#define IMI_MASK_NODE_INPUT     0x01
#define IMI_MASK_NODE_OUTPUT    0x02
#define IMI_MASK_NODE_GRAPH     0x04
#define IMI_MASK_NODE_ANY       0xff
// @brief:  show graph input/output tensor info
// @param:  quant[in]   quant or not(maybe we can get from graph)
// @param:  mask[in]    0x01: input; 0x02: output; 0x04: graph
static int imi_utils_tm_show_graph(const graph_t graph, char quant, char mask) {
    node_t node;
    int count, idx, rc = 0;

    if (mask & IMI_MASK_NODE_INPUT) {
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
    }

    if (mask & IMI_MASK_NODE_OUTPUT) {
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
    }

    if (mask & IMI_MASK_NODE_GRAPH) {
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
    }

    fprintf(stdout, "\n");
    if (rc) {
        fprintf(stderr, "[Error] %d dims are <= 0\n", rc);
    }
    return rc;
}

#endif // !__IMI_UTILS_TM_DEBUG_H__
