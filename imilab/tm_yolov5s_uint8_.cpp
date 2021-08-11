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

//#define CROSS_VALIDATION
//#define _DEBUG

/* std c includes */
#include <stdio.h>
#include <stdlib.h>
/* std c++ includes */
#include <vector>
/* imilab includes */
#include "utils/imi_utils_object.hpp"
#include "utils/imi_utils_visual.hpp"
#include "utils/imi_utils_tm.h"   // for: imi_utils_tm_run_graph
#include "utils/imi_utils_elog.h" // for: log_xxxx
#include "utils/imi_utils_tm_debug.h"
#include "utils/imi_model_yolov5.hpp"

// postprocess threshold
static float prob_threshold = 0.4f; // 0.25f
static float nms_threshold = 0.40f; // 0.45f

// example models for show usage
static const char* models[] = {
    "yolov5s.v5.uint8.tmfile", // official model
    "yolov5s.uint8.tmfile",    // imilab model
};

// time cost
static double start_time = 0.;

// @brief:  yolov5 output tensor postprocess
// P3 node[0].output[0]: (1, 3, 80, 80, 85), stride=640/80=8 ,  small obj
// P4 node[1].output[0]: (1, 3, 40, 40, 85), stride=640/40=16, middle obj
// P5 node[2].output[0]: (1, 3, 20, 20, 85), stride=640/20=32,  large obj
// @param:  model[in]   input yolo model info
// @param:  graph[in]   input yolo graph inst
// @param:  quant[in]   output quant info
// @param:  proposals   output detected boxes
static int proposals_objects_get(const yolov3& model,
                                 graph_t& graph, const void* param[], std::vector<Object>& proposals)
{
    proposals.clear();

    static const void* buffer[NODE_CNT_YOLOV5S] = {0};
    for (int i = 0; i < NODE_CNT_YOLOV5S; i++)
    {
        tm_quant_t quant = (tm_quant_t)param[i];
        uint8_t* data_u8 = (uint8_t*)quant->buffer;
        if (NULL == buffer[i]) buffer[i] = calloc(quant->size, sizeof(float));
        float* data_fp32 = (float*)buffer[i];
        for (int c = 0; c < quant->size; c++)
        {
            data_fp32[c] = ((float)data_u8[c] - quant->zero_point) * quant->scale;
        }
    }

    /* generate output proposals */
    return imi_utils_yolov3_proposals_generate(model, buffer, proposals, prob_threshold);
}

int main(int argc, char* argv[])
{
    int repeat_count = 1;
    int num_thread = 1;
    int target_class = -1;

    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* output_file = "output.rgb";

    yolov3& model = yolov5s;
    image input = make_empty_image(640, 360, 3);

    int ret, frame = 1, fc = 0;
    while ((ret = getopt(argc, argv, "c:m:i:r:t:w:h:n:o:f:s:u")) != -1)
    {
        switch (ret)
        {
        case 'c':
            target_class = atoi(optarg);
            if (target_class < 0 || model.class_num <= target_class)
            {
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
    if (nullptr == model_file || nullptr == image_file)
    {
        log_error("Tengine model or image file not specified!\n");
        show_usage(argv[0], models);
        return -1;
    }
    if (!check_file_exist(model_file) || !check_file_exist(image_file))
    {
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 0;

    /* inital tengine */
    if (0 != (ret = init_tengine()))
    {
        log_error("Initial tengine failed.\n");
        return -1;
    }
    log_echo("tengine-lite library version: %s\n", get_tengine_version());

    // cache start time
    start_time = get_current_time();

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph)
    {
        log_error("Load model to graph failed\n");
        return -1;
    }

    /* get input tensor of graph */
    tensor_t tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == tensor)
    {
        log_error("Get input tensor failed\n");
        return -1;
    }

    /* get shape of input tensor */
    int i, dims[DIM_NUM]; // nchw
    int dim_num = get_tensor_shape(tensor, dims, DIM_NUM);
    log_echo("input tensor shape: %d(", dim_num);
    for (i = 0; i < dim_num; i++)
    {
        log_echo(" %d", dims[i]);
    }
    log_echo(")\n");
    if (DIM_NUM != dim_num)
    {
        log_error("Get input tensor shape error\n");
        return -1;
    }
    if (12 == dims[DIM_IDX_C])
    {
        // revert from focus shape to origin image shape
        dims[DIM_IDX_W] *= 2, dims[DIM_IDX_H] *= 2, dims[DIM_IDX_C] /= 4;
    }
    else if (3 == dims[DIM_IDX_C])
    {
        // reset input shape as focus shape
        int _dims[DIM_NUM] = {dims[0], dims[1] * 4, dims[2] / 2, dims[3] / 2};
        if (set_tensor_shape(tensor, _dims, DIM_NUM) < 0)
        {
            log_error("Set input tensor shape failed\n");
            return -1;
        }
    }
    else
    {
        log_error("Unavailable channel number: %d\n", dims[DIM_IDX_C]);
        return -1;
    }

    image& lb = model.lb;
    lb = make_image(dims[DIM_IDX_W], dims[DIM_IDX_H], dims[DIM_IDX_C]);
    int img_size = lb.w * lb.h * lb.c;
    /* set the data mem to input tensor */
    if (set_tensor_buffer(tensor, lb.data, img_size /* * sizeof(float)*/) < 0)
    {
        log_error("Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph to infer shape, and set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        log_error("Prerun multithread graph failed.\n");
        return -1;
    }
    //imi_utils_tm_show_graph(graph, 0, IMI_MASK_NODE_OUTPUT);

    /* get input parameter info */
    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(tensor, &input_scale, &input_zero_point, 1);

    /* get output parameter info */
    tm_quant quant_param[NODE_CNT_YOLOV5S] = {0};
    const void* buffer[NODE_CNT_YOLOV5S] = {&quant_param[0], &quant_param[1], &quant_param[2]};
    if (imi_utils_yolov3_get_output_parameter(graph, buffer, NODE_CNT_YOLOV5S, 1) < 0)
    {
        log_error("get output parameter failed\n");
        return -1;
    }

    int bgr = -1;
    std::vector<Object> proposals;
    std::vector<Object> objects;

    FILE* fout = output_file ? fopen(output_file, "wb") : NULL;
    FILE* fp = fopen(image_file, "rb");

    // load raw data(non-planar)
    if (strstr(image_file, "bgra"))
        input.c = 4, bgr = 1;
    else if (strstr(image_file, "bgr"))
        input.c = 3, bgr = 1;
    else if (strstr(image_file, "rgba"))
        input.c = 4, bgr = 0;
    else if (strstr(image_file, "rgb"))
        input.c = 3, bgr = 0;
    else
    {
        log_error("unknown test data format!\n");
        goto exit;
    }
    input.data = (float*)calloc(sizeof(float), input.c * input.w * input.h);

read_data:
    /* prepare process input data, set the data mem to input tensor */
    if (1 != (ret = imi_utils_yolov5_load_data(fp, input, bgr, lb, (const float(*)[3])model.usr_data, input_scale, input_zero_point)))
    {
        log_error("%s\n", ret ? "get_input_data error!" : "read input data fin");
        goto exit;
    }
    fc++;
#ifdef CROSS_VALIDATION
    {
        FILE* fp1 = fopen("temp_.dat", "rb");
        fread(lb.data, 1, img_size, fp1);
        fclose(fp1);
    }
#endif // CROSS_VALIDATION
    log_echo("======================================\n");
    log_echo("Frame No.%03d:\n", fc);

    /* run graph */
    if (imi_utils_tm_run_graph(graph, repeat_count) < 0)
    {
        goto exit;
    }

    /* process the detection result */
    if (proposals_objects_get(model, graph, buffer, proposals) < 0)
    {
        goto exit;
    }

    // sort all proposals by score from highest to lowest
    imi_utils_objects_qsort(proposals);
    // filter objects
    objects = imi_utils_proposals_filter(proposals, Size2i(input.w, input.h), Size2i(lb.w, lb.h), nms_threshold);

    // draw objects
    imi_utils_objects_draw(objects, input, target_class, model.class_names);

    // save result to output
    if (fout)
    {
        imi_utils_image_save_permute_chw2hwc(fout, input, bgr);
    }

    if (fc < frame) goto read_data;

exit:
    if (fp) fclose(fp);
    if (fout) fclose(fout);
    free_image(input);
    free_image(lb);
    if (1 < fc) log_echo("total frame: %d\n", fc);
    log_echo("total time cost: %.2f s\n", (get_current_time() - start_time) / 1000.);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    /* show test status */
    imi_utils_tm_run_status(NULL);
    return 0;
}
