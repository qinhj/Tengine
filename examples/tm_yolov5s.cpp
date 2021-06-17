/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: xwwang@openailab.com
 * Author: stevenwudi@fiture.com
 * Author: qinhongjie@imilab.com
 *
 * original models:
 *  https://github.com/ultralytics/yolov3
 *  https://github.com/ultralytics/yolov5
 */

/* std c includes */
#include <stdio.h>
#include <stdlib.h>
/* std c++ includes */
#include <vector>
/* tengine includes */
#include "tengine/c_api.h"
/* examples includes */
#include "common.h"
#include "tengine_operations.h"

#define DIM_NUM     4
#define DIM_IDX_N   0
#define DIM_IDX_C   1
#define DIM_IDX_H   2
#define DIM_IDX_W   3

/* #include "utils/imi_utils_types.hpp" */

template<typename _Tp> class Rect_ {
public:
    typedef _Tp value_type;

    //! default constructor
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);

    //! area (width*height) of the rectangle
    _Tp area() const;

    _Tp x; //!< x coordinate of the top-left corner
    _Tp y; //!< y coordinate of the top-left corner
    _Tp width; //!< width of the rectangle
    _Tp height; //!< height of the rectangle
};

typedef Rect_<float> Rect2f;

template<typename _Tp> inline
Rect_<_Tp>::Rect_()
    : x(0), y(0), width(0), height(0) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    : x(_x), y(_y), width(_width), height(_height) {}

template<typename _Tp> static inline
Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    Rect_<_Tp> c = a;
    return c &= b;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator &= (Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp x1 = std::max(a.x, b.x);
    _Tp y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if (a.width <= 0 || a.height <= 0)
        a = Rect_<_Tp>();
    return a;
}

template<typename _Tp> inline
_Tp Rect_<_Tp>::area() const {
    const _Tp result = width * height;
    return result;
}

template<typename _Tp> class Size_ {
public:
    typedef _Tp value_type;

    //! default constructor
    Size_();
    Size_(_Tp _width, _Tp _height);

    _Tp width; //!< the width
    _Tp height; //!< the height
};

typedef Size_<int> Size2i;

template<typename _Tp> inline
Size_<_Tp>::Size_()
    : width(0), height(0) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(_Tp _width, _Tp _height)
    : width(_width), height(_height) {}

/* #include "utils/imi_utils_object.hpp" */

typedef struct object_s {
    Rect2f rect;
    int label;
    float prob;
} Object;

// @brief:  qsort_descent_inplace
// @note:   _Tp must have member "prob"
template<typename _Tp>
static void qsort_descent_inplace(std::vector<_Tp> &objects, int left, int right) {
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

// @brief:  qsort_descent_inplace
// @note:   _Tp must have member "prob"
template<typename _Tp>
static void imi_utils_objects_qsort(std::vector<_Tp> &objects, char descent = 1) {
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// @brief:  nms sorted bboxes
// @note:   _Tp must have member "rect"
template<typename _Tp>
static void imi_utils_objects_nms(const std::vector<_Tp>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        const _Tp &obj = objects[i];
        areas[i] = obj.rect.area();
    }

    for (int i = 0; i < n; i++) {
        const _Tp &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size() && keep; j++) {
            const _Tp &b = objects[picked[j]];

            // intersection over union
            float inter_area = (a.rect & b.rect).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area / union_area;
            // float IoU = inter_area / union_area
            if (iou > nms_threshold) {
                keep = 0;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
}

// @brief:  filter proposal objects
template<typename _Tp>
static std::vector<_Tp> imi_utils_proposals_filter(const std::vector<_Tp> &proposals,
    const Size2i &image_size, const Size2i &letter_box, float nms_threshold = 0.45f) {
    std::vector<int> picked;
    // apply nms with nms_threshold
    imi_utils_objects_nms(proposals, picked, nms_threshold);

    int count = picked.size();

    // post process: scale and offset for letter box
    Size2i lb_offset;
    float lb_scale = -1;
    if (0 < letter_box.width && 0 < letter_box.height) {
        float scale_w = letter_box.width * 1.0 / image_size.width;
        float scale_h = letter_box.height * 1.0 / image_size.height;
        lb_scale = scale_h < scale_w ? scale_h : scale_w;
        lb_offset.width = int(lb_scale * image_size.width);
        lb_offset.height = int(lb_scale * image_size.height);
        lb_offset.width = (letter_box.width - lb_offset.width) / 2;
        lb_offset.height = (letter_box.height - lb_offset.height) / 2;
    }

    std::vector<_Tp> objects(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        // post process: from letter box to input image
        if (0 < letter_box.width && 0 < letter_box.height) {
            x0 = (x0 - lb_offset.width) / lb_scale;
            y0 = (y0 - lb_offset.height) / lb_scale;
            x1 = (x1 - lb_offset.width) / lb_scale;
            y1 = (y1 - lb_offset.height) / lb_scale;
        }

        x0 = std::max(std::min(x0, (float)(image_size.width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(image_size.height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(image_size.width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(image_size.height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return objects;
}

/* #include "utils/imi_utils_visual.hpp" */

// @param:  cls[in] target class(-1: all)
template<typename _Tp>
int imi_utils_objects_draw(const std::vector<_Tp> &objects, image &img, const char * const *labels) {
    size_t size = objects.size();
    fprintf(stdout, "detected objects num: %zu\n", size);

    for (size_t i = 0; i < size; i++) {
        const _Tp &obj = objects[i];
        if (labels) {
            fprintf(stdout, "[%2d]: %3.3f%%, [(%7.3f, %7.3f), (%7.3f, %7.3f)], %s\n",
                obj.label, obj.prob * 100, obj.rect.x, obj.rect.y,
                obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, labels[obj.label]);
        }
        draw_box(img, obj.rect.x, obj.rect.y,
            obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, 2, 0, 255, 0);
    }
    return 0;
}

/* #include "utils/imi_utils_coco.h" */

static const float coco_mean[3] = { 0, 0, 0 };
static const float coco_scale[3] = { 0.003921, 0.003921, 0.003921 };
static const char *coco_class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",            // 00-04
    "train", "truck", "boat", "traffic light",                              // 05-09
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",          // 10-14
    "cat", "dog", "horse", "sheep", "cow",                                  // 15-19
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",         // 20-24
    "handbag", "tie", "suitcase", "frisbee",                                // 25-29
    "skis", "snowboard", "sports ball", "kite", "baseball bat",             // 30-34
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", // 35-39
    "wine glass", "cup", "fork", "knife", "spoon",                          // 40-44
    "bowl", "banana", "apple", "sandwich", "orange",                        // 45-49
    "broccoli", "carrot", "hot dog", "pizza", "donut",                      // 50-54
    "cake", "chair", "couch", "potted plant", "bed",                        // 55-59
    "dining table", "toilet", "tv", "laptop", "mouse",                      // 60-64
    "remote", "keyboard", "cell phone", "microwave", "oven",                // 65-69
    "toaster", "sink", "refrigerator", "book", "clock",                     // 70-74
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"            // 75-79
};
static const int coco_class_num = sizeof(coco_class_names) / sizeof(coco_class_names[0]);

/* #include "utils/imi_utils_yolo.h" */

// yolov3/yolov5* strides && anchors
static const int strides[] = { 8, 16, 32 };
static const int anchors[] = {
    10, 13, 16, 30, 33, 23,     // P3/8
    30, 61, 62, 45, 59, 119,    // P4/16
    116, 90, 156, 198, 373, 326,// P5/32
};
// yolov3-tiny strides && anchors
static const int strides_tiny[] = { 16, 32 };
static const int anchors_tiny[] = {
    10, 14, 23, 27, 37, 58,     // P4/16
    81, 82, 135, 169, 344, 319, // P5/32
};

// yolo model struct
typedef struct yolo_s {
    /* inputs/outputs */
    image lb;       // letter box size(as input shape)
    int outputs;    // output node count
    int class_num;
    const char* const *class_names;
    /* model settings */
    const int *strides;
    int anchors_num;
    const int *anchors;
} yolo;

// yolov3 model instance
static yolo yolov3 = {
    make_empty_image(640, 640, 3), 3,
    coco_class_num, coco_class_names,
    strides, 3, anchors
};
// yolov3-tiny model info
static yolo yolov3_tiny = {
    make_empty_image(640, 640, 3), 2,
    coco_class_num, coco_class_names,
    strides_tiny, 3, anchors_tiny
};

/* #include "utils/imi_utils_yolov3.hpp" */

// @brief:  get output buffer address
// @param:  graph[in]   input runtime graph
// @param:  param[out]  buffer list if 0 == quant, else tm_quant_t
// @param:  count[in]   output buffer count
// @param:  quant[in]   quant(uint8) or not
static int imi_utils_yolov3_get_output_parameter(
    const graph_t &graph, const void **param, int count, char quant) {
    // check inputs
    if (NULL == param || get_graph_output_node_number(graph) != count) {
        fprintf(stderr, "invalid input params: param=%p, %d=?%d\n",
            param, count, get_graph_output_node_number(graph));
        return -1;
    }

    tensor_t tensor = NULL;
    for (int i = 0; i < count && 0 == quant; i++) {
        /* get output tensor info */
        tensor = get_graph_output_tensor(graph, i, 0);
        param[i] = tensor ? get_tensor_buffer(tensor) : NULL;
        if (NULL == param[i]) {
            fprintf(stderr, "get tensor[%d] buffer NULL\n", i);
            return -1;
        }
    }
    for (int i = 0; i < count && 0 != quant; i++) {
        /* get output tensor info */
        tensor = get_graph_output_tensor(graph, i, 0);
        /* get output quant info */
    }
    return 0;
}

static __inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

// @brief:  generate proposals
// @param:  model[in]   yolov3 model info
// @param:  data[in]    output tensor buffer(xywh, box_score, {cls_score} x cls_num)
// @param:  objects[out]
// @param:  threshold
static int imi_utils_yolov3_proposals_generate(const yolo *model,
    const void *data[], std::vector<Object> &objects, float threshold) {
    int out_num = 4 + 1 + model->class_num; // rent, score, class_num

    // walk through all output nodes
    for (int i = 0; i < model->outputs; i++) {
        const float *feat = (const float *)data[i];
        int feat_w = (model->lb).w / model->strides[i];
        int feat_h = (model->lb).h / model->strides[i];
        // 1 x anchors_num x feature_map_w x feature_map_h x out_num
        for (int a = 0; a < model->anchors_num; a++) {
            float anchor_w = model->anchors[i * model->anchors_num * 2 + a * 2 + 0];
            float anchor_h = model->anchors[i * model->anchors_num * 2 + a * 2 + 1];
            // todo: check ahwo or awho plz
            for (int h = 0; h < feat_h; h++) {
                for (int w = 0; w < feat_w; w++) {
                    int loc_idx = (a * feat_w * feat_h + h * feat_w + w) * out_num;
                    // process class score
                    int cls_idx = 0;
                    float cls_score = feat[loc_idx + 5 + cls_idx];
                    for (int s = 1; s < model->class_num; s++) {
                        if (cls_score < feat[loc_idx + 5 + s]) {
                            cls_idx = s, cls_score = feat[loc_idx + 5 + s];
                        }
                    }
                    // process box score
                    float box_score = feat[loc_idx + 4];
                    float prob = sigmoid(box_score) * sigmoid(cls_score);
                    if (threshold < prob) {
                        float dx = sigmoid(feat[loc_idx + 0]);
                        float dy = sigmoid(feat[loc_idx + 1]);
                        float dw = sigmoid(feat[loc_idx + 2]);
                        float dh = sigmoid(feat[loc_idx + 3]);
                        float pred_cx = (dx * 2.0f - 0.5f + w) * model->strides[i];
                        float pred_cy = (dy * 2.0f - 0.5f + h) * model->strides[i];
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

/* #include "utils/imi_utils_yolov5.hpp" */

static yolo &yolov5 = yolov3;
static int imi_utils_yolov5_focus_data(const float *data, image &lb) {
    // check inputs
    if (NULL == data || NULL == lb.data) {
        return -1;
    }

    for (int i = 0; i < 2; i++) {       // corresponding to rows
        for (int g = 0; g < 2; g++) {   // corresponding to cols
            for (int c = 0; c < lb.c; c++) {
                for (int h = 0; h < lb.h / 2; h++) {
                    for (int w = 0; w < lb.w / 2; w++) {
                        int in_index =
                            i + g * lb.w + c * lb.w * lb.h +
                            h * 2 * lb.w + w * 2;
                        int out_index =
                            i * 2 * lb.c * (lb.w / 2) * (lb.h / 2) +
                            g * lb.c * (lb.w / 2) * (lb.h / 2) +
                            c * (lb.w / 2) * (lb.h / 2) +
                            h * (lb.w / 2) +
                            w;

                        lb.data[out_index] = data[in_index];
                    }
                }
            }
        }
    }
    return 0;
}

/* yolov5s examples */

#define NODE_OUT_MAX    4   // yolov5+

// example models for show usage
static const char *models[] = {
    "yolov5s.v5", "yolov3.v9.5", "yolov3-tiny.v9.5",
};
static void show_usage(const char *exe, const char *model[], int num) {
    fprintf(stdout, "[Usage]:  [-h]\n");
    fprintf(stdout, "    [-m model_file] [-i input_file] [-o output_file]\n");
    fprintf(stdout, "    [-r repeat_count] [-t thread_count] [-s threshold]\n\n");
    fprintf(stdout, "[Examples]:\n");
    for (int i = 0; i < num; i++) {
        fprintf(stdout, "   %s -m %s.tmfile -i ssd_dog.jpg -o ssd_dog_%s.jpg -t4 -r100\n",
            exe, model[i], model[i]);
    }
};

// postprocess threshold
static float prob_threshold = 0.25f;
static float nms_threshold = 0.45f;
// time cost
static double start_time = 0.;

// @brief:  yolov5 output tensor postprocess
// P3 node[0].output[0]: (1, 3, 80, 80, 85), stride=640/80=8 ,  small obj
// P4 node[1].output[0]: (1, 3, 40, 40, 85), stride=640/40=16, middle obj
// P5 node[2].output[0]: (1, 3, 20, 20, 85), stride=640/20=32,  large obj
// @param:  model[in]   input yolo model info
// @param:  graph[in]   input yolo graph inst
// @param:  buffer[in]  output tensor buffer
// @param:  proposals   output detected boxes
static int proposals_objects_get(const yolo *model,
    graph_t &graph, const void *buffer[], std::vector<Object> &proposals) {
    proposals.clear();

    /* generate output proposals */
    return imi_utils_yolov3_proposals_generate(model, buffer, proposals, prob_threshold);
}

int main(int argc, char* argv[]) {
    int repeat_count = 1;
    int num_thread = 1;

    const char *model_file = nullptr;
    const char *image_file = nullptr;
    const char *output_file = "yolo_out";

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:o:s:")) != -1) {
        switch (res) {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'r':
            repeat_count = std::strtoul(optarg, nullptr, 10);
            break;
        case 't':
            num_thread = std::strtoul(optarg, nullptr, 10);
            break;
        case 'h':
            show_usage(argv[0], models, sizeof(models) / sizeof(models[0]));
            return 0;
        case 'o':
            output_file = optarg;
            break;
        case 's':
            prob_threshold = (float)atof(optarg);
            break;
        default:
            break;
        }
    }

    /* check files */
    if (nullptr == model_file || nullptr == image_file) {
        fprintf(stderr, "Tengine model or image file not specified!\n");
        show_usage(argv[0], models, sizeof(models) / sizeof(models[0]));
        return -1;
    }
    if (!check_file_exist(model_file) || !check_file_exist(image_file)) {
        return -1;
    }

    // parse yolo type by name
    yolo *model = NULL;
    if (strstr(model_file, "yolov5s")) model = &yolov5;
    else if (strstr(model_file, "yolov3")) {
        model = strstr(model_file, "tiny") ? &yolov3_tiny : &yolov3;
    }
    else {
        fprintf(stderr, "unknown yolo model type!\n");
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    // start time
    start_time = get_current_time();

    /* inital tengine */
    int ret = init_tengine();
    if (0 != ret) {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stdout, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph) {
        fprintf(stderr, "Load model to graph failed\n");
        return -1;
    }

    /* get input tensor of graph */
    tensor_t tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == tensor) {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    /* get shape of input tensor */
    int i, dims[DIM_NUM]; // nchw
    int dim_num = get_tensor_shape(tensor, dims, DIM_NUM);
    fprintf(stdout, "input tensor shape: %d(", dim_num);
    for (i = 0; i < dim_num; i++) {
        fprintf(stdout, " %d", dims[i]);
    }
    fprintf(stdout, ")\n");
    if (DIM_NUM != dim_num) {
        fprintf(stderr, "Get input tensor shape error\n");
        return -1;
    }
    // check input shape
    if (12 == dims[DIM_IDX_C] && strstr(model_file, "yolov5s")) {
        // revert from focus shape to origin image shape
        dims[DIM_IDX_W] *= 2, dims[DIM_IDX_H] *= 2, dims[DIM_IDX_C] /= 4;
    }
    else if (3 == dims[DIM_IDX_C] && strstr(model_file, "yolov5s")) {
        fprintf(stdout, "[Warn] Invalid focus shape!!!\n");
        fprintf(stdout, "Is this tmfile optimized by old yolov5s-opt.py(before 2021/06/10)?\n");
        // reset input shape as focus shape
        int _dims[DIM_NUM] = { dims[0], dims[1] * 4, dims[2] / 2, dims[3] / 2 };
        if (set_tensor_shape(tensor, _dims, DIM_NUM) < 0) {
            fprintf(stderr, "Set input tensor shape failed\n");
            return -1;
        }
    }
    else if (3 == dims[DIM_IDX_C] && strstr(model_file, "yolov3")) {
    }
    else {
        fprintf(stderr, "Unavailable channel number: %d\n", dims[DIM_IDX_C]);
        return -1;
    }

    // create letterbox
    image &lb = model->lb;
    lb = make_image(dims[DIM_IDX_W], dims[DIM_IDX_H], dims[DIM_IDX_C]);
    int img_size = lb.w * lb.h * lb.c;
    /* set the data mem to input tensor */
    if (set_tensor_buffer(tensor, lb.data, img_size * sizeof(float)) < 0) {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph to infer shape, and set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0) {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* get output parameter info */
    const void *buffer[NODE_OUT_MAX] = { 0 };
    if (imi_utils_yolov3_get_output_parameter(graph, buffer, model->outputs, 0) < 0) {
        fprintf(stderr, "get output parameter failed\n");
        return -1;
    }

    std::vector<Object> proposals;
    std::vector<Object> objects;
    // load encoded image
    image input = imread(image_file);

    /* prepare process input data, set the data mem to input tensor */
    float lb_scale_w = (float)lb.w / input.w;
    float lb_scale_h = (float)lb.h / input.h;
    int im_rw = lb_scale_w < lb_scale_h ? lb.w : input.w * lb_scale_h;
    int im_rh = lb_scale_w < lb_scale_h ? input.h * lb_scale_w : lb.h;
    if (im_rw != input.w || im_rh != input.h) {
        // resize to fit letter box
        image resized = resize_image(input, im_rw, im_rh);
        fprintf(stdout, "resize image from (%d,%d) to (%d,%d)\n", input.w, input.h, im_rw, im_rh);
        // attach resized image to letter box
        add_image(resized, lb, (lb.w - im_rw) / 2, (lb.h - im_rh) / 2);
        // release resized image
        free_image(resized);
    }
    else {
        // attach image to letter box directly
        add_image(input, lb, (lb.w - im_rw) / 2, (lb.h - im_rh) / 2);
    }
    // focus operator: (C, H, W) -> (C*4, H/2, W/2) for yolov5s
    float *_data = strstr(model_file, "yolov5s") ? (float *)malloc(img_size * sizeof(float)) : lb.data;
    for (int idx, k = 0; k < lb.c; k++) {
        for (int i = 0; i < lb.h; i++) {
            for (int j = 0; j < lb.w; j++) {
               idx = k * lb.h * lb.w + i * lb.w + j;
                _data[idx] = (lb.data[idx] - coco_mean[k]) * coco_scale[k];
            }
        }
    }
    if (strstr(model_file, "yolov5s")) {
        imi_utils_yolov5_focus_data(_data, lb);
        free(_data);
    }

    /* run graph */
    double min_time = DBL_MAX, max_time = DBL_MIN, run_graph_time = 0.;
    for (int i = 0; i < repeat_count; i++) {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0) {
            fprintf(stderr, "Run graph failed\n");
            goto exit;
        }
        double cur = get_current_time() - start;
        run_graph_time += cur;
        min_time = min_time < cur ? min_time : cur;
        max_time = max_time < cur ? cur : max_time;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            run_graph_time/repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    if (proposals_objects_get(model, graph, buffer, proposals) < 0) {
        goto exit;
    }

    // sort all proposals by score from highest to lowest
    imi_utils_objects_qsort(proposals);
    // filter objects
    objects = imi_utils_proposals_filter(proposals, Size2i(input.w, input.h), Size2i(lb.w, lb.h), nms_threshold);

    // draw objects
    imi_utils_objects_draw(objects, input, model->class_names);

    // save result to output
    save_image(input, output_file);

exit:
    free_image(input);
    free_image(lb);
    fprintf(stdout, "total time cost: %.2f s\n", (get_current_time() - start_time) / 1000.);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
    return 0;
}
