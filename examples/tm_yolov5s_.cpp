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

/* std c includes */
#include <stdlib.h>
/* std c++ includes */
#include <vector>
#include <algorithm>
#include <cmath>
/* opencv includes */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 /* imilab includes */
#include "imilab/imi_utils_tm.h"// imi_utils_tm_run_graph

struct Size2i {
    int width;
    int height;
};

struct Point2f {
    float x;
    float y;
};

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// allow none square letterbox, set default letterbox size
static const int letterbox_rows = 640;
static const int letterbox_cols = 640;
// postprocess threshold
static const float prob_threshold = 0.25f;
static const float nms_threshold = 0.45f;

static Size2i scale_tmp = { 0, 0 };
static Point2f scale_ratio = { 0., 0. };

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


static void generate_proposals(int stride, const float* feat, float prob_threshold, std::vector<Object>& objects,
    int letterbox_cols, int letterbox_rows) {
    static float anchors[18] = { 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 };

    int anchor_num = 3;
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int cls_num = 80;
    int anchor_group;
    if (stride == 8)
        anchor_group = 1;
    if (stride == 16)
        anchor_group = 2;
    if (stride == 32)
        anchor_group = 3;
    for (int h = 0; h <= feat_h - 1; h++) {
        for (int w = 0; w <= feat_w - 1; w++) {
            for (int a = 0; a <= anchor_num - 1; a++) {
                //process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++) {
                    float score = feat[a * feat_w * feat_h * 85 + h * feat_w * 85 + w * 85 + s + 5];
                    if (score > class_score) {
                        class_index = s;
                        class_score = score;
                    }
                }
                //process box score
                float box_score = feat[a * feat_w * feat_h * 85 + (h * feat_w) * 85 + w * 85 + 4];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold) {
                    int loc_idx = a * feat_h * feat_w * 85 + h * feat_w * 85 + w * 85;
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
                    objects.push_back(obj);
                }
            }
        }
    }
}

static void draw_objects(const std::vector<Object>& objects, const cv::Mat& bgr) {
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
            obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

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

    cv::imwrite("yolov5_out.jpg", image);
}

static void show_usage() {
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

void get_input_data_focus(const char* image_file, float* input_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale) {
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols)) {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0.5 / scale[0] + mean[0], 0.5 / scale[1] + mean[1], 0.5 / scale[2] + mean[2]));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    img_new.convertTo(img_new, CV_32FC3);
    float* img_data = (float*)img_new.data;
    float* input_temp = (float*)malloc(3 * letterbox_cols * letterbox_rows * sizeof(float));

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++) {
        for (int w = 0; w < letterbox_cols; w++) {
            for (int c = 0; c < 3; c++) {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_temp[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }

    /* focus process */
    for (int i = 0; i < 2; i++) // corresponding to rows
    {
        for (int g = 0; g < 2; g++) // corresponding to cols
        {
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < letterbox_rows / 2; h++) {
                    for (int w = 0; w < letterbox_cols / 2; w++) {
                        int in_index = i + g * letterbox_cols + c * letterbox_cols * letterbox_rows +
                            h * 2 * letterbox_cols + w * 2;
                        int out_index = i * 2 * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) +
                            g * 3 * (letterbox_cols / 2) * (letterbox_rows / 2) +
                            c * (letterbox_cols / 2) * (letterbox_rows / 2) +
                            h * (letterbox_cols / 2) +
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

// yolov5 postprocess
// 0: 1, 3, 20, 20, 85
// 1: 1, 3, 40, 40, 85
// 2: 1, 3, 80, 80, 85
static int proposals_objects_get(graph_t &graph, std::vector<Object> &proposals) {
    int stride = 32;
    float *p_data = NULL;
    tensor_t p_tensor = NULL;
    for (int i = 3; 0 < i; i--) {
        p_tensor = get_graph_output_tensor(graph, i - 1, 0);
        p_data = p_tensor ? (float *)get_tensor_buffer(p_tensor) : NULL;
        if (NULL == p_data) {
            fprintf(stderr, "[%s] get_tensor_buffer NULL\n", __FUNCTION__);
            return -1;
        }
        generate_proposals(stride, p_data, prob_threshold, proposals, letterbox_cols, letterbox_rows);
        stride >>= 1;
    }
    return 0;
}

static int get_letterbox_scale_ratio(const Size2i &image_size, Size2i &tmp, Point2f &ratio) {
    float scale_letterbox;
    if ((letterbox_rows * 1.0 / image_size.height) < (letterbox_cols * 1.0 / image_size.width)) {
        scale_letterbox = letterbox_rows * 1.0 / image_size.height;
    }
    else {
        scale_letterbox = letterbox_cols * 1.0 / image_size.width;
    }
    tmp.width = int(scale_letterbox * image_size.width);
    tmp.height = int(scale_letterbox * image_size.height);

    ratio.x = (float)image_size.height / tmp.height;
    ratio.y = (float)image_size.width / tmp.width;

    tmp.height = (letterbox_rows - tmp.height) / 2;
    tmp.width = (letterbox_cols - tmp.width) / 2;
    return 0;
}

static std::vector<Object> proposals_objects_filter(const std::vector<Object> &proposals, const Size2i &image_size) {
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    fprintf(stdout, "%d objects are picked\n", count);

    std::vector<Object> objects(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = (x0 - scale_tmp.width) * scale_ratio.x;
        y0 = (y0 - scale_tmp.height) * scale_ratio.y;
        x1 = (x1 - scale_tmp.width) * scale_ratio.x;
        y1 = (y1 - scale_tmp.height) * scale_ratio.y;

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

int main(int argc, char* argv[]) {
    const char *model_file = nullptr;
    const char *image_file = nullptr;

    int img_c = 3;
    const float mean[3] = { 0, 0, 0 };
    const float scale[3] = { 0.003921, 0.003921, 0.003921 };

    int repeat_count = 1;
    int num_thread = 1;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1) {
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

    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }
    Size2i image_size = { img.cols, img.rows };
    get_letterbox_scale_ratio(image_size, scale_tmp, scale_ratio);

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

    int img_size = letterbox_rows * letterbox_cols * img_c;
    /* set the input shape to initial the graph, and pre-run graph to infer shape */
    int dims[] = { 1, 12, int(letterbox_rows / 2), int(letterbox_cols / 2) };
    float *input_data = (float *)malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == input_tensor) {
        fprintf(stderr, "[%s] Get input tensor failed\n", __FUNCTION__);
        return -1;
    }

    if (0 != set_tensor_shape(input_tensor, dims, 4)) {
        fprintf(stderr, "[%s] Set input tensor shape failed\n", __FUNCTION__);
        return -1;
    }

    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0) {
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

    /* prepare process input data, set the data mem to input tensor */
    get_input_data_focus(image_file, input_data, letterbox_rows, letterbox_cols, mean, scale);

    /* run graph */
    if (imi_utils_tm_run_graph(graph, repeat_count) < 0) {
        ;//goto exit;
    }

    /* process the detection result */
    if (proposals_objects_get(graph, proposals) < 0) {
        ;//goto exit;
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    // filter objects
    objects = proposals_objects_filter(proposals, image_size);
    // draw objects
    draw_objects(objects, img);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
