// ============================================================
//                  Imilab Model: Yolov3 APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_MODEL_YOLOV3_HPP__
#define __IMI_MODEL_YOLOV3_HPP__

/* std c++ includes */
#include <cmath> // for: exp
/* tengine includes */
#include "tengine/c_api.h"      // for: graph_t
#include "tengine_operations.h" // for: image
/* imilab includes */
#include "imi_utils_elog.h" // for: log_xxxx
#include "imi_utils_coco.h" // for: coco_class_names
#include "imi_utils_object.hpp"
#include "imi_utils_tm_quant.h" // for: tm_quant_t
#if defined(_DEBUG)             //&& 0
#include "imi_utils_tm_debug.h" // for: _imi_utils_tm_show_tensor
#endif                          // _DEBUG

// output node count
#define NODE_CNT_YOLOV3      3
#define NODE_CNT_YOLOV3_TINY 2

// yolov3 strides
static const int strides[] = {8, 16, 32};
// yolov3-tiny strides
static const int strides_tiny[] = {16, 32};

// yolov3 anchors
static const int anchors[] = {
    10, 13, 16, 30, 33, 23,      // P3/8
    30, 61, 62, 45, 59, 119,     // P4/16
    116, 90, 156, 198, 373, 326, // P5/32
};
// yolov3-tiny anchors
static const int anchors_tiny[] = {
    10, 14, 23, 27, 37, 58,     // P4/16
    81, 82, 135, 169, 344, 319, // P5/32
};

typedef struct yolov3_s
{
    /* input/output */
    image lb;      // letter box size
    int outputs;   // output node count
    int class_num; // default: 80
    const char* const* class_names;
    /* hyp */
    const int* strides;
    int anchors_num; // default: 3
    const int* anchors;
    const void* usr_data;
} yolov3;

// yolov3 model info
static yolov3 yolov3_std = {
    make_empty_image(608, 608, 3),
    NODE_CNT_YOLOV3,
    coco_class_num,
    coco_class_names,
    /* hyp */
    strides,
    3,
    anchors,
    coco_image_cov,
};
// yolov3-tiny model info
static yolov3 yolov3_tiny = {
    make_empty_image(416, 416, 3),
    NODE_CNT_YOLOV3_TINY,
    coco_class_num,
    coco_class_names,
    /* hyp */
    strides_tiny,
    3,
    anchors_tiny,
    coco_image_cov,
};

// @brief:  get output buffer address
// @param:  graph[in]   input runtime graph
// @param:  param[out]  buffer list if 0 == quant, else tm_quant_t
// @param:  count[in]   output buffer count
// @param:  quant[in]   quant(uint8) or not
static int imi_utils_yolov3_get_output_parameter(
    const graph_t& graph, const void** param, int count, char quant)
{
    // check inputs
    if (NULL == param || get_graph_output_node_number(graph) != count)
    {
        log_error("invalid input params: param=%p, %d=?%d\n",
                  param, count, get_graph_output_node_number(graph));
        return -1;
    }

    tensor_t tensor = NULL;
    for (int i = 0; i < count && 0 == quant; i++)
    {
        /* get output tensor info */
        tensor = get_graph_output_tensor(graph, i, 0);
        param[i] = tensor ? get_tensor_buffer(tensor) : NULL;
        if (NULL == param[i])
        {
            log_error("get tensor[%d] buffer NULL\n", i);
            return -1;
        }
#if defined(_DEBUG) //&& 0
        _imi_utils_tm_show_tensor(tensor, i, 0, 1);
#endif // _DEBUG
    }
    for (int i = 0; i < count && 0 != quant; i++)
    {
        /* get output tensor info */
        tensor = get_graph_output_tensor(graph, i, 0);
        /* get output quant info */
        tm_quant_t quant = (tm_quant_t)param[i];
        if (tensor && NULL == quant) quant = (tm_quant_t)calloc(1, sizeof(tm_quant));
        if (0 != imi_utils_tm_quant_get(tensor, quant))
        {
            log_error("get tensor[%d] quant info failed\n", i);
            return -1;
        }
#if defined(_DEBUG) //&& 0
        _imi_utils_tm_show_tensor(tensor, i, 1, 1);
#endif // _DEBUG
    }
    return 0;
}

static __inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
// @brief:  generate proposals
// @param:  model[in]   yolov3 model info
// @param:  data[in]    output tensor buffer(xywh, box_score, {cls_score} x cls_num)
// @param:  objects[out]
// @param:  threshold
static int imi_utils_yolov3_proposals_generate(const yolov3& model,
                                               const void* data[], std::vector<Object>& objects, float threshold)
{
    int out_num = 4 + 1 + model.class_num; // rent, score, class_num

    // walk through all output nodes
    for (int i = 0; i < model.outputs; i++)
    {
        const float* feat = (const float*)data[i];
        int feat_w = (model.lb).w / model.strides[i];
        int feat_h = (model.lb).h / model.strides[i];
        // 1 x anchors_num x feature_map_w x feature_map_h x out_num
        for (int a = 0; a < model.anchors_num; a++)
        {
            int anchor_w = model.anchors[i * model.anchors_num * 2 + a * 2 + 0];
            int anchor_h = model.anchors[i * model.anchors_num * 2 + a * 2 + 1];
            // todo: check ahwo or awho plz
            for (int h = 0; h < feat_h; h++)
            {
                for (int w = 0; w < feat_w; w++)
                {
                    int loc_idx = (a * feat_w * feat_h + h * feat_w + w) * out_num;
                    // process class score
                    int cls_idx = 0;
                    float cls_score = feat[loc_idx + 5 + cls_idx];
                    for (int s = 1; s < model.class_num; s++)
                    {
                        if (cls_score < feat[loc_idx + 5 + s])
                        {
                            cls_idx = s, cls_score = feat[loc_idx + 5 + s];
                        }
                    }
                    // process box score
                    float box_score = feat[loc_idx + 4];
                    float prob = sigmoid(box_score) * sigmoid(cls_score);
                    if (threshold < prob)
                    {
                        float dx = sigmoid(feat[loc_idx + 0]);
                        float dy = sigmoid(feat[loc_idx + 1]);
                        float dw = sigmoid(feat[loc_idx + 2]);
                        float dh = sigmoid(feat[loc_idx + 3]);
                        float pred_cx = (dx * 2.0f - 0.5f + w) * model.strides[i];
                        float pred_cy = (dy * 2.0f - 0.5f + h) * model.strides[i];
                        float pred_w = dw * dw * 4.0f * anchor_w;
                        float pred_h = dh * dh * 4.0f * anchor_h;
                        // top left
                        float x0 = pred_cx - pred_w * 0.5f;
                        float y0 = pred_cy - pred_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = pred_w;
                        obj.rect.height = pred_h;
                        obj.label = cls_idx;
                        obj.prob = prob;
                        objects.push_back(obj);
                    }
                }
            }
        }
    }

    return objects.size();
}

#endif // !__IMI_MODEL_YOLOV3_HPP__
