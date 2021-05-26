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

//#define USE_OPENCV
//#define _DEBUG

/* std c includes */
#include <stdio.h>
#include <stdlib.h>
/* std c++ includes */
#include <vector>
/* imilab includes */
#include "imilab/imi_utils_object.hpp"
#include "imilab/imi_utils_visual.hpp"
#include "imilab/imi_utils_yolov5.hpp"
#include "imilab/imi_utils_tm.h"    // for: imi_utils_tm_run_graph
#include "imilab/imi_utils_elog.h"  // for: log_xxxx
#include "imilab/imi_utils_tm_debug.h"

// postprocess threshold
static float prob_threshold = 0.6f; // 0.25f
static float nms_threshold = 0.40f; // 0.45f

// example models for show usage
static const char *models[] = {
    "yolov5s.v5.tmfile", // official model
    "yolov5s.tmfile",    // imilab model
};

#ifdef USE_OPENCV

// load input images
static void get_input_data_focus(const char *image_file, image &lb) {
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    //Size2i resize;
    int resize_rows;
    int resize_cols;
    if ((lb.h * 1.0 / img.rows) < (lb.w * 1.0 / img.cols)) {
        scale_letterbox = lb.h * 1.0 / img.rows;
    }
    else {
        scale_letterbox = lb.w * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    // resize to part of letter box(e.g. 506x381 -> 640x481)
    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    //cv::imwrite("yolov5_resize.jpg", img);
    img.convertTo(img, CV_32FC3);
    //cv::imwrite("yolov5_resize_32FC3.jpg", img);

    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(lb.w, lb.h, CV_32FC3,
        cv::Scalar(0.5 / coco_image_cov[1][0] + coco_image_cov[0][0], 0.5 / coco_image_cov[1][1] + coco_image_cov[0][1], 0.5 / coco_image_cov[1][2] + coco_image_cov[0][2]));
    int top = (lb.h - resize_rows) / 2;
    int bot = (lb.h - resize_rows + 1) / 2;
    int left = (lb.w - resize_cols) / 2;
    int right = (lb.w - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    //cv::imwrite("yolov5_make_border.jpg", img_new);

    img_new.convertTo(img_new, CV_32FC3);
    //cv::imwrite("yolov5_make_border_32FC3.jpg", img_new);

    float* img_data = (float*)img_new.data;
    float* input_temp = (float*)malloc(3 * lb.w * lb.h * sizeof(float));

    /* nhwc to nchw */
    for (int h = 0; h < lb.h; h++) {
        for (int w = 0; w < lb.w; w++) {
            for (int c = 0; c < 3; c++) {
                int in_index = h * lb.w * 3 + w * 3 + c;
                int out_index = c * lb.w * lb.h + h * lb.w + w;
                input_temp[out_index] = (img_data[in_index] - coco_image_cov[0][c]) * coco_image_cov[1][c];
            }
        }
    }

    /* focus process: 3x640x640 -> 12x320x320 */
    imi_utils_yolov5_focus_data(input_temp, lb);
    free(input_temp);
}

#endif // USE_OPENCV

// @brief:  yolov5 output tensor postprocess
// P3 node[0].output[0]: (1, 3, 80, 80, 85), stride=640/80=8 ,  small obj
// P4 node[1].output[0]: (1, 3, 40, 40, 85), stride=640/40=16, middle obj
// P5 node[2].output[0]: (1, 3, 20, 20, 85), stride=640/20=32,  large obj
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

    yolov3 &model = yolov5s;
    // reset letter box size if necessary
    model.lb = make_image(640, 640, 3);
    // allow none square letterbox, set default letterbox size
    image &lb = model.lb;
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

    /* set the input shape to initial the graph, and pre-run graph to infer shape */
    int dims[] = { 1, 12, lb.h / 2, lb.w / 2 };

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == input_tensor) {
        log_error("Get input tensor failed\n");
        return -1;
    }

    if (0 != set_tensor_shape(input_tensor, dims, 4)) {
        log_error("Set input tensor shape failed\n");
        return -1;
    }

    int img_size = lb.c * lb.h * lb.w, bgr = 0;
    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, lb.data, img_size * sizeof(float)) < 0) {
        log_error("Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0) {
        log_error("Prerun multithread graph failed.\n");
        return -1;
    }
    //imi_utils_tm_show_graph(graph, 0, IMI_MASK_NODE_OUTPUT);

    /* get output parameter info */
    const void *buffer[NODE_CNT_YOLOV5S] = { 0 };
    if (imi_utils_yolov3_get_output_parameter(graph, buffer, NODE_CNT_YOLOV5S, 0) < 0) {
        log_error("get output parameter failed\n");
        return -1;
    }

    std::vector<Object> proposals;
    std::vector<Object> objects;

#ifdef USE_OPENCV
    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty()) {
        log_error("cv::imread %s failed\n", image_file);
        return -1;
    }
    input.w = img.cols, input.h = img.rows;
#else // !USE_OPENCV
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
#endif // USE_OPENCV

read_data:
    /* prepare process input data, set the data mem to input tensor */
#ifdef USE_OPENCV
    get_input_data_focus(image_file, lb);
#else // !USE_OPENCV
    if (1 != (ret = imi_utils_yolov5_load_data(fp, input, bgr, lb, (const float (*)[3])model.usr_data, -1, 0))) {
        log_error("%s\n", ret ? "get_input_data error!" : "read input data fin");
        goto exit;
    }
    fc++;
#endif // USE_OPENCV

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
#ifdef USE_OPENCV
    imi_utils_objects_draw(objects, img, target_class, model.class_names, output_file);
exit:
#else // !USE_OPENCV
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

#endif // USE_OPENCV
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
