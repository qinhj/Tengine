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
 * Copyright (c) 2021, OPEN AI LAB && IMILAB
 * Author: xwwang@openailab.com
 * Author: stevenwudi@fiture.com
 * Author: qinhongjie@imilab.com
 */

//#define _DEBUG

/* It seems that enable customer shape can speed up a little(1~2ms in average). */
#define ENABLE_CUSTOMER_SHAPE
#ifdef ENABLE_CUSTOMER_SHAPE
#ifndef DFLT_TENSOR_SHAPE
#define DFLT_TENSOR_SHAPE   1,3,640,640 // nchw
#endif // DFLT_TENSOR_SHAPE
#ifndef INPUT_TENSOR_SHAPE
#define INPUT_TENSOR_SHAPE  DFLT_TENSOR_SHAPE
#endif // !INPUT_TENSOR_SHAPE
#endif // ENABLE_CUSTOMER_SHAPE

/* std c includes */
#include <stdio.h>
#include <stdlib.h>
/* std c++ includes */
#include <vector>
/* imilab includes */
#include "utils/imi_utils_object.hpp"
#include "utils/imi_utils_visual.hpp"
#include "utils/imi_utils_yolov3.hpp"
#include "utils/imi_utils_tm.h"     // for: imi_utils_tm_run_graph
#include "utils/imi_utils_elog.h"   // for: log_xxxx
#include "utils/imi_utils_image.h"  // for: imi_utils_image_load_letterbox
#include "utils/imi_utils_tm_debug.h"

// postprocess threshold
static float prob_threshold = 0.4f; // 0.25f
static float nms_threshold = 0.40f; // 0.45f

// example models for show usage
static const char *models[] = {
    "yolov3-tiny.v9.5.tmfile", // official model
    "yolov3-tiny.tmfile",      // imilab model
};

// @brief:  yolov3-tiny output tensor postprocess
// P4 node[0].output[0]: (1, 3, 26, 26, 5+nc), stride=416/26=16, middle obj
// P5 node[1].output[0]: (1, 3, 13, 13, 5+nc), stride=416/13=32,  large obj
// @param:  model[in]   input yolo model info
// @param:  graph[in]   input yolo graph inst
// @param:  buffer[in]  output tensor buffer
// @param:  proposals   output detected boxes
static int proposals_objects_get(const yolov3 &model,
    graph_t &graph, const void *buffer[], std::vector<Object> &proposals) {
    proposals.clear();

    /* generate output proposals */
    return imi_utils_yolov3_proposals_generate(model, buffer, proposals, prob_threshold);
}

int main(int argc, char* argv[]) {
    int repeat_count = 1;
    int num_thread = 1;
    int target_class = -1;

    const char *model_file = nullptr;
    const char *image_file = nullptr;
    const char *output_file = "output.rgb";

    yolov3 &model = yolov3_tiny;
    image input = make_empty_image(640, 360, 3);

    int res, frame = 1, fc = 0;
    while ((res = getopt(argc, argv, "c:m:i:r:t:w:h:n:o:f:s:u")) != -1) {
        switch (res) {
        case 'c':
            target_class = atoi(optarg);
            if (target_class < 0 || model.class_num <= target_class) {
                // reset all invalid argument as -1
                target_class = -1;
            }
            break;
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
        case 'n':
            model.class_num = atoi(optarg);
            break;
        case 'w':
            input.w = atoi(optarg);
            break;
        case 'h':
            input.h = atoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'f':
            frame = atoi(optarg);
            break;
        case 's':
            prob_threshold = (float)atof(optarg);
            break;
        case 'u':
            show_usage(argv[0], models);
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (nullptr == model_file || nullptr == image_file) {
        log_error("Tengine model or image file not specified!\n");
        show_usage(argv[0], models);
        return -1;
    }
    if (!check_file_exist(model_file) || !check_file_exist(image_file)) {
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    int ret = init_tengine();
    if (0 != ret) {
        log_error("Initial tengine failed.\n");
        return -1;
    }
    log_echo("tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph) {
        log_error("Load model to graph failed\n");
        return -1;
    }

    /* get input tensor of graph */
    tensor_t tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == tensor) {
        log_error("Get input tensor failed\n");
        return -1;
    }

#ifndef ENABLE_CUSTOMER_SHAPE
    /* get shape of input tensor */
    int i, dims[DIM_NUM]; // nchw
    int dim_num = get_tensor_shape(tensor, dims, DIM_NUM);
    log_echo("input tensor shape: %d(", dim_num);
    for (i = 0; i < dim_num; i++) {
        log_echo(" %d", dims[i]);
    }
    log_echo(")\n");
    if (DIM_NUM != dim_num) {
        log_error("Get input tensor shape error\n");
        return -1;
    }
#else // customer shape
    int i, dim_num;
    /* set input tensor shape (if necessary) */
    int dims[DIM_NUM] = { INPUT_TENSOR_SHAPE };
    if (0 != set_tensor_shape(tensor, dims, DIM_NUM)) {
        log_error("Set input tensor shape failed\n");
        return -1;
    }
#endif // !ENABLE_CUSTOMER_SHAPE

    image &lb = model.lb;
    lb = make_image(dims[DIM_IDX_W], dims[DIM_IDX_H], dims[DIM_IDX_C]);
    int img_size = lb.w * lb.h * lb.c;
    /* set the data mem to input tensor */
    if (set_tensor_buffer(tensor, lb.data, img_size * sizeof(float)) < 0) {
        log_error("Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph to infer shape, and set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0) {
        log_error("Prerun multithread graph failed.\n");
        return -1;
    }
    imi_utils_tm_show_graph(graph, 0, IMI_MASK_NODE_OUTPUT);

    /* get output parameter info */
    const void *buffer[NODE_CNT_YOLOV3_TINY] = { 0 };
    if (imi_utils_yolov3_get_output_parameter(graph, buffer, NODE_CNT_YOLOV3_TINY, 0) < 0) {
        log_error("get output parameter failed\n");
        return -1;
    }

    int bgr = -1;
    std::vector<Object> proposals;
    std::vector<Object> objects;

    FILE *fout = output_file ? fopen(output_file, "wb") : NULL;
    FILE *fp = fopen(image_file, "rb");

    // load raw data(non-planar)
    if (strstr(image_file, "bgra")) input.c = 4, bgr = 1;
    else if (strstr(image_file, "bgr")) input.c = 3, bgr = 1;
    else if (strstr(image_file, "rgba")) input.c = 4, bgr = 0;
    else if (strstr(image_file, "rgb")) input.c = 3, bgr = 0;
    else {
        log_error("unknown test data format!\n");
        goto exit;
    }
    input.data = (float *)calloc(sizeof(float), input.c * input.w * input.h);

read_data:
    /* prepare process input data, set the data mem to input tensor */
    if (1 != (ret = imi_utils_image_load_letterbox(fp, input, bgr, lb, (const float (*)[3])model.usr_data))) {
        log_error("%s\n", ret ? "get_input_data error!" : "read input data fin");
        goto exit;
    }
    fc++;

    /* run graph */
    if (imi_utils_tm_run_graph(graph, repeat_count) < 0) {
        goto exit;
    }

    /* process the detection result */
    if (proposals_objects_get(model, graph, buffer, proposals) < 0) {
        goto exit;
    }

    // sort all proposals by score from highest to lowest
    imi_utils_objects_qsort(proposals);
    // filter objects
    objects = imi_utils_proposals_filter(proposals, Size2i(input.w, input.h), Size2i(lb.w, lb.h), nms_threshold);

    // draw objects
    imi_utils_objects_draw(objects, input, target_class, model.class_names);

    // save result to output
    if (fout) {
        imi_utils_image_save_permute_chw2hwc(fout, input, bgr);
    }

    if (fc < frame) goto read_data;

exit:
    if (fp) fclose(fp);
    if (fout) fclose(fout);
    free_image(input);
    free_image(lb);
    if (1 < fc) log_echo("total frame: %d\n", fc);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    /* show test status */
    imi_utils_tm_run_status(NULL);
    return 0;
}
