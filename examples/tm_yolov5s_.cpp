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
 */

//#define USE_OPENCV
//#define _DEBUG

/* std c includes */
#include <stdlib.h>
/* std c++ includes */
#include <vector>
#include <cmath>    // for: exp
/* imilab includes */
#include "imilab/imi_utils_elog.h"
#ifdef USE_OPENCV
/* opencv includes */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#else // !USE_OPENCV
#include "imilab/imi_utils_types.hpp"
#include "imilab/imi_imread.h"
#endif // USE_OPENCV
#include "imilab/imi_utils_coco.h"  // for: coco_class_names
#include "imilab/imi_utils_sort.hpp"
#include "imilab/imi_utils_tm.h"    // for: imi_utils_tm_run_graph

#ifdef USE_OPENCV
using namespace cv;
#endif // USE_OPENCV

struct Object {
    Rect2f rect;
    int label;
    float prob;
};

static int class_num = coco_class_num;
static const char **class_names = coco_class_names;
// postprocess threshold
static const float prob_threshold = 0.6f; // 0.25f
static const float nms_threshold = 0.40f; // 0.45f

// allow none square letterbox, set default letterbox size
static image lb = make_image(640, 640, 3);
static const float cov[][3] = {
    { 0, 0, 0 }, // mean
    { 0.003921, 0.003921, 0.003921 } // scale
};

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

// todo: optimize
// @param:  feat[in]    anchor results: box.x, box.y, box.w, box.h, box.score, {cls.score} x cls.num
static void generate_proposals(int stride, const float *feat, float prob_threshold, std::vector<Object>& objects) {
    static float anchors[18] = { 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 };

    int cls_num = class_num; // 80
    int out_num = 4 + 1 + cls_num;// rent, score, cls_num

    int anchor_num = 3;
    int anchor_group;
    if (stride == 8)
        anchor_group = 1;
    if (stride == 16)
        anchor_group = 2;
    if (stride == 32)
        anchor_group = 3;

    int feat_w = lb.w / stride;
    int feat_h = lb.h / stride;
    for (int h = 0; h < feat_h; h++) {
        for (int w = 0; w < feat_w; w++) {
            for (int a = 0; a < anchor_num; a++) {
                int loc_idx = a * feat_w * feat_h * out_num + h * feat_w * out_num + w * out_num;
                // process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++) {
                    float score = feat[loc_idx + 5 + s];
                    if (score > class_score) {
                        class_index = s;
                        class_score = score;
                    }
                }
                // process box score
                float box_score = feat[loc_idx + 4];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold) {
                    float dx = sigmoid(feat[loc_idx + 0]);
                    float dy = sigmoid(feat[loc_idx + 1]);
                    float dw = sigmoid(feat[loc_idx + 2]);
                    float dh = sigmoid(feat[loc_idx + 3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    /*if (0 == class_index)*/ objects.push_back(obj);
                }
            }
        }
    }
}

#ifdef USE_OPENCV

static void draw_objects(const std::vector<Object>& objects, const cv::Mat& bgr, int cls) {
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
            obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        if (-1 != cls && obj.label != cls) continue;

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 0, 0));
    }

    cv::imwrite("yolov5s_out.jpg", image);
}

// load input images
static void get_input_data_focus(const char *image_file, float *input_data) {
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
        cv::Scalar(0.5 / cov[1][0] + cov[0][0], 0.5 / cov[1][1] + cov[0][1], 0.5 / cov[1][2] + cov[0][2]));
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
                input_temp[out_index] = (img_data[in_index] - cov[0][c]) * cov[1][c];
            }
        }
    }

    /* focus process: 3x640x640 -> 12x320x320 */
    /*
     | 0 2 |          C0-0, C1-0, C2-0,
     | 1 3 | x C3 =>  C0-1, C1-1, C2-1, x C12
                      C0-2, C1-2, C2-2,
                      C0-3, C1-3, C2-3,
    */
    for (int i = 0; i < 2; i++) {       // corresponding to rows
        for (int g = 0; g < 2; g++) {   // corresponding to cols
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < lb.h / 2; h++) {
                    for (int w = 0; w < lb.w / 2; w++) {
                        int in_index = i + g * lb.w + c * lb.w * lb.h +
                            h * 2 * lb.w + w * 2;
                        int out_index = i * 2 * 3 * (lb.w / 2) * (lb.h / 2) +
                            g * 3 * (lb.w / 2) * (lb.h / 2) +
                            c * (lb.w / 2) * (lb.h / 2) +
                            h * (lb.w / 2) +
                            w;

                        /* quant to uint8 */
                        input_data[out_index] = input_temp[in_index];
                    }
                }
            }
        }
    }

    free(input_temp);
}

#else // !USE_OPENCV

static int draw_objects(const std::vector<Object>& objects, image &img, int cls) {
    size_t size = objects.size();
    fprintf(stdout, "detected objects num: %zu\n", size);

    for (size_t i = 0; i < size; i++) {
        const Object& obj = objects[i];
        fprintf(stdout, "[%2d]: %3.3f%%, [(%4.0f, %4.0f), (%4.0f, %4.0f)], %s\n",
            obj.label, obj.prob * 100, obj.rect.x, obj.rect.y,
            obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        if (-1 == cls || obj.label == cls) {
            draw_box(img, obj.rect.x, obj.rect.y,
                obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, 2, 0, 255, 0);
        }
    }
    return 0;
}

#endif // USE_OPENCV

// yolov5 postprocess
// 0: 1, 3, 20, 20, 85
// 1: 1, 3, 40, 40, 85
// 2: 1, 3, 80, 80, 85
static int proposals_objects_get(graph_t &graph, std::vector<Object> &proposals) {
    proposals.clear();
    int stride = 32;
    float *p_data = NULL;
    tensor_t p_tensor = NULL;
    for (int i = 3; 0 < i; i--) {
        // ==================================================================
        // ========== This part is to get tensor information ================
        // ==================================================================
        p_tensor = get_graph_output_tensor(graph, i - 1, 0);
        p_data = p_tensor ? (float *)get_tensor_buffer(p_tensor) : NULL;
        if (NULL == p_data) {
            fprintf(stderr, "[%s] get_tensor_buffer NULL\n", __FUNCTION__);
            return -1;
        }
        generate_proposals(stride, p_data, prob_threshold, proposals);
        stride >>= 1;
    }
    return 0;
}

static std::vector<Object> proposals_objects_filter(const std::vector<Object> &proposals, const Size2i &image_size) {
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    fprintf(stdout, "%d objects are picked\n", count);

    // post process: scale and offset
    float lb_scale;
    if ((lb.h * 1.0 / image_size.height) < (lb.w * 1.0 / image_size.width)) {
        lb_scale = lb.h * 1.0 / image_size.height;
    }
    else {
        lb_scale = lb.w * 1.0 / image_size.width;
    }
    Size2i off = { int(lb_scale * image_size.width), int(lb_scale * image_size.height) };
    off.width = (lb.w - off.width) / 2;
    off.height = (lb.h - off.height) / 2;
    log_debug("[%s] lb scale: %3.4f, off: (%d, %d)\n", __FUNCTION__, lb_scale, off.width, off.height);

    std::vector<Object> objects(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        // post process: from letter box to input image
        x0 = (x0 - off.width) / lb_scale;
        y0 = (y0 - off.height) / lb_scale;
        x1 = (x1 - off.width) / lb_scale;
        y1 = (y1 - off.height) / lb_scale;

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

static void show_usage() {
    fprintf(stdout, "[Usage]:  [-u]\n");
    fprintf(stdout, "    [-m model_file] [-i input_file] [-o output_file] [-n class_number] [-c target_class]\n");
    fprintf(stdout, "    [-w width] [-h height] [-f max_frame] [-r repeat_count] [-t thread_count]\n");
    fprintf(stdout, "[Examples]:\n");
    fprintf(stdout, "   # coco 80 classes\n");
    fprintf(stdout, "   tm_yolov5s_ -m yolov5s.v5.tmfile -i /Dataset/imilab_640x360x3_bgr_catdog.rgb24 -o imilab_640x360x3_bgr_catdog.rgb24 -f 200\n");
    fprintf(stdout, "   # specific class of coco 80 classes(e.g. person)\n");
    fprintf(stdout, "   tm_yolov5s_ -m yolov5s.v5.tmfile -i /Dataset/imilab_640x360x3_bgr_human1.rgb24 -o imilab_640x360x3_bgr_human1.rgb24 -c 0 -f 100\n");
    fprintf(stdout, "   tm_yolov5s_ -m yolov5s.v5.tmfile -i /Dataset/imilab_640x360x3_bgr_human2.rgb24 -o imilab_640x360x3_bgr_human2.rgb24 -c 0 -f 500\n");
    fprintf(stdout, "   # single class(e.g. person)\n");
    fprintf(stdout, "   tm_yolov5s_ -m yolov5s.tmfile -i /Dataset/imilab_640x360x3_bgr_catdog.rgb24 -o imilab_640x360x3_bgr_catdog.rgb24 -n 1 -f 200\n");
    fprintf(stdout, "   tm_yolov5s_ -m yolov5s.tmfile -i /Dataset/imilab_640x360x3_bgr_human1.rgb24 -o imilab_640x360x3_bgr_human1.rgb24 -n 1 -f 100\n");
    fprintf(stdout, "   tm_yolov5s_ -m yolov5s.tmfile -i /Dataset/imilab_640x360x3_bgr_human2.rgb24 -o imilab_640x360x3_bgr_human2.rgb24 -n 1 -f 500\n");
}

int main(int argc, char* argv[]) {
    int repeat_count = 1;
    int num_thread = 1;
    int target_class = -1;

    const char *model_file = nullptr;
    const char *image_file = nullptr;
    const char *output_file = "output.rgb";
    image input = make_empty_image(640, 360, 3);

    int res, frame = 1, fc = 0;
    while ((res = getopt(argc, argv, "c:m:i:r:t:w:h:n:o:f:u")) != -1) {
        switch (res) {
        case 'c':
            target_class = atoi(optarg);
            if (target_class < 0 || class_num <= target_class) {
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
            class_num = atoi(optarg);
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
        case 'u':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (nullptr == model_file || nullptr == image_file) {
        fprintf(stderr, "[%s] Error: Tengine model or image file not specified!\n", __FUNCTION__);
        show_usage();
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
        fprintf(stderr, "[%s] Initial tengine failed.\n", __FUNCTION__);
        return -1;
    }
    fprintf(stdout, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph) {
        fprintf(stderr, "[%s] Load model to graph failed: %d\n", __FUNCTION__, get_tengine_errno());
        return -1;
    }

    /* set the input shape to initial the graph, and pre-run graph to infer shape */
    int dims[] = { 1, 12, lb.h / 2, lb.w / 2 };

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == input_tensor) {
        fprintf(stderr, "[%s] Get input tensor failed\n", __FUNCTION__);
        return -1;
    }

    if (0 != set_tensor_shape(input_tensor, dims, 4)) {
        fprintf(stderr, "[%s] Set input tensor shape failed\n", __FUNCTION__);
        return -1;
    }

    int img_size = lb.c * lb.h * lb.w, bgr = 0;
    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, lb.data, img_size * sizeof(float)) < 0) {
        fprintf(stderr, "[%s] Set input tensor buffer failed\n", __FUNCTION__);
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0) {
        fprintf(stderr, "[%s] Prerun multithread graph failed.\n", __FUNCTION__);
        return -1;
    }

    std::vector<Object> proposals;
    std::vector<Object> objects;

#ifdef USE_OPENCV
    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty()) {
        fprintf(stderr, "[%s] cv::imread %s failed\n", __FUNCTION__, image_file);
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
        fprintf(stderr, "[%s] unknown test data format!\n", __FUNCTION__);
        goto exit;
    }
    input.data = (float *)calloc(sizeof(float), input.c * input.w * input.h);
#endif // USE_OPENCV

read_data:
    /* prepare process input data, set the data mem to input tensor */
#ifdef USE_OPENCV
    get_input_data_focus(image_file, lb.data);
#else // !USE_OPENCV
    if (1 != (ret = get_input_data_yolov5(fp, lb, bgr, input, cov))) {
        fprintf(stderr, "%s\n", ret ? "get_input_data error!" : "read input data fin");
        goto exit;
    }
    fc++;
#endif // USE_OPENCV

    /* run graph */
    if (imi_utils_tm_run_graph(graph, repeat_count) < 0) {
        goto exit;
    }

    /* process the detection result */
    if (proposals_objects_get(graph, proposals) < 0) {
        goto exit;
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    // filter objects
    objects = proposals_objects_filter(proposals, Size2i(input.w, input.h));

    // draw objects
#ifdef USE_OPENCV
    draw_objects(objects, img, target_class);
exit:
#else // !USE_OPENCV
    draw_objects(objects, input, target_class);

    // save result to output
    if (fout) {
        unsigned char uc[3];
        int img_size_ = input.w * input.h;
        for (int i = 0; i < img_size_; i++) {
            uc[0] = (unsigned char)(*(input.data + i + 2 * img_size_)); // b
            uc[1] = (unsigned char)(*(input.data + i + 1 * img_size_)); // g
            uc[2] = (unsigned char)(*(input.data + i + 0 * img_size_)); // r
            fwrite(uc, sizeof(unsigned char), 3, fout);
        }
    }

    if (fc < frame) goto read_data;

exit:
    fclose(fp);
    if (fout) fclose(fout);
    free(input.data);
    printf("total frame: %d\n", fc);
#endif // USE_OPENCV
    free(lb.data);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
