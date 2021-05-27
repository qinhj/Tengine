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
 * Copyright (c) 2020, OPEN AI LAB && IMILAB
 * Author: qtang@openailab.com
 * Author: qinhongjie@imilab.com
 */

/* It's hard to say whether use letter box could have better results or not. */
//#define TRY_LETTER_BOX

/* It seems that enable customer shape can speed up a little(1~2ms in average). */
//#define ENABLE_CUSTOMER_SHAPE
#ifdef ENABLE_CUSTOMER_SHAPE
#ifndef DFLT_TENSOR_SHAPE
#define DFLT_TENSOR_SHAPE   1,3,300,300 // nchw
#endif // DFLT_TENSOR_SHAPE
#ifndef INPUT_TENSOR_SHAPE
#define INPUT_TENSOR_SHAPE  DFLT_TENSOR_SHAPE // 1,3,360,640
#endif // !INPUT_TENSOR_SHAPE
#endif // ENABLE_CUSTOMER_SHAPE

/* std c includes */
#include <stdio.h>  // for: fprintf
#include <string.h> // for: strstr
/* imilab includes */
#include "utils/imi_utils_tm.h"     // for: imi_utils_tm_run_graph
#include "utils/imi_utils_voc.h"    // for: voc_class_names, ...
#include "utils/imi_utils_elog.h"   // for: log_xxxx
#include "utils/imi_utils_image.h"  // for: imi_utils_image_load_bgr
#include "utils/imi_utils_tm_debug.h"

typedef struct Box {
    float x0, y0;
    float x1, y1;
    int class_idx;
    float score;
} Box_t;

static const char* const *class_names = voc_class_names;

// example models for show usage
static const char *models[] = {
    "mobilenet_ssd.tmfile",     // online model
    "mobilenet_ssd.imi.tmfile", // imilab model
};

#ifdef TRY_LETTER_BOX
static float lb_scale = 0.f;    // letter box scale
static int lb_ow = 0, lb_oh = 0;// letter box offset
#endif // TRY_LETTER_BOX
static image lb;

// @return: box count
static int post_process_ssd(image im, float threshold, const void *data, int num, int tc) {
    const float *outdata = (const float *)data;

    Box_t box;
    int i, cnt = 0;
    int im_w = im.w, im_h = im.h; //im_w *= (640/300), im_h *= (360/300);
    for (i = 0; i < num; i++) {
        if (threshold <= outdata[1]) {
            box.class_idx = outdata[0];
            box.score = outdata[1];
#ifndef TRY_LETTER_BOX
            box.x0 = outdata[2] < 0 ? 0 : 1 < outdata[2] ? im_w : outdata[2] * im_w;
            box.y0 = outdata[3] < 0 ? 0 : 1 < outdata[3] ? im_h : outdata[3] * im_h;
            box.x1 = outdata[4] < 0 ? 0 : 1 < outdata[4] ? im_w : outdata[4] * im_w;
            box.y1 = outdata[5] < 0 ? 0 : 1 < outdata[5] ? im_h : outdata[5] * im_h;
#else // TRY_LETTER_BOX
            box.x0 = (outdata[2] * lb.w - lb_ow) / lb_scale;
            box.y0 = (outdata[3] * lb.h - lb_oh) / lb_scale;
            box.x1 = (outdata[4] * lb.w - lb_ow) / lb_scale;
            box.y1 = (outdata[5] * lb.h - lb_oh) / lb_scale;
            box.x0 = box.x0 < 0 ? 0 : im_w < box.x0 ? im_w : box.x0;
            box.y0 = box.y0 < 0 ? 0 : im_h < box.y0 ? im_h : box.y0;
            box.x1 = box.x1 < 0 ? 0 : im_w < box.x1 ? im_w : box.x1;
            box.y1 = box.y1 < 0 ? 0 : im_h < box.y1 ? im_h : box.y1;
#endif // !TRY_LETTER_BOX
            // draw box to image
            if (tc < 0 || tc == box.class_idx) {
                draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
            }

            log_echo("score: %.3f%%, box: (%.3f, %.3f) (%.3f, %.3f), label: %s\n",
                box.score * 100, box.x0, box.y0, box.x1, box.y1, class_names[box.class_idx]);
            cnt++;
        }
        outdata += 6;
    }
    log_echo("detect box num: %d, select box num: %d\n", num, cnt);
    return cnt;
}

int main(int argc, char* argv[]) {
    int repeat_count = 1;
    int num_thread = 1;
    int target_class = -1;
    int class_num = voc_class_num;
    float threshold = 0.5f;

    const char *model_file = NULL;
    const char *image_file = NULL;
    const char *output_file = "output.rgb";

    image im = make_empty_image(640, 360, 3);

    int res, frame = 1, fc = 0;
    while ((res = getopt(argc, argv, "c:m:i:r:t:w:h:n:o:f:s:u")) != -1) {
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
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'n':
            class_num = atoi(optarg);
            break;
        case 'w':
            im.w = atoi(optarg);
            break;
        case 'h':
            im.h = atoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'f':
            frame = atoi(optarg);
            break;
        case 's':
            threshold = (float)atof(optarg);
            break;
        case 'u':
            show_usage(argv[0], models);
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (NULL == model_file || NULL == image_file) {
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
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (NULL == graph) {
        log_error("Load model to graph failed\n");
        return -1;
    }

    /* get input tensor of graph */
    tensor_t tensor = get_graph_input_tensor(graph, 0, 0);
    if (NULL == tensor) {
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
    if (set_tensor_shape(tensor, dims, DIM_NUM) < 0) {
        log_error("Set input tensor shape failed\n");
        return -1;
    }
#endif // !ENABLE_CUSTOMER_SHAPE

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
    //imi_utils_tm_show_graph(graph, 0, IMI_MASK_NODE_OUTPUT);

    /* get output tensor of graph */
    tensor = get_graph_output_tensor(graph, 0, 0);
    if (NULL == tensor) {
        log_error("Get output tensor failed\n");
        return -1;
    }
#if 0 // it seems that get output tensor shape could cost more time during run graph
    /* get output tensor shape: (1, 100, 6, 1) */
    dim_num = get_tensor_shape(tensor, dims, DIM_NUM);
    if (dim_num < 2) {
        log_error("Get output tensor shape error\n");
        return -1;
    }
    log_echo("maximum output box num: %d\n", dims[1]);
#endif

    // to support read/load video.rgb or video.bgr
    int bgr = -1;
    FILE *fp = NULL, *fout = NULL;
    if (strstr(image_file, "bgra")) im.c = 4, bgr = 0;
    else if (strstr(image_file, "bgr")) im.c = 3, bgr = 0;
    else if (strstr(image_file, "rgba")) im.c = 4, bgr = 1;
    else if (strstr(image_file, "rgb")) im.c = 3, bgr = 1;
    else { ; }
    if (-1 != bgr) {
        fp = fopen(image_file, "rb");
        fout = fopen(output_file, "wb");
    }
    im = fp ? make_image(im.w, im.h, im.c) : imread(image_file);

#ifdef TRY_LETTER_BOX
    float lb_scale_w = (float)lb.w / im.w;
    float lb_scale_h = (float)lb.h / im.h;
    int im_rw = lb_scale_w < lb_scale_h ? lb.w : im.w * lb_scale_h;
    int im_rh = lb_scale_w < lb_scale_h ? im.h * lb_scale_w : lb.h;
    lb_scale = lb_scale_w < lb_scale_h ? lb_scale_w : lb_scale_h;
    lb_ow = (lb.w - im_rw) / 2;
    lb_oh = (lb.h - im_rh) / 2;
#endif // TRY_LETTER_BOX

read_data:
    /* prepare process input data, set the data mem to input tensor */
    if (fp) {
        // load raw data from file and convert to bgr planar format
        if (1 != (ret = imi_utils_image_load_bgr(fp, im, bgr, lb.c))) {
            fprintf(stderr, "%s\n", ret ? "get_input_data error!" : "read input data fin");
            goto exit;
        }
        fc++;
        log_echo("======================================\n");
        log_echo("Frame No.%03d:\n", fc);

        int i, j, k, idx;
        const float *_data = im.data;
#ifndef TRY_LETTER_BOX
        // strategy: resize to letter box directly
        if (im.w != lb.w || im.h != lb.h) {
            // resize to network input shape
            tengine_resize_f32(im.data, lb.data, lb.w, lb.h, im.c, im.h, im.w);
            _data = lb.data;
        }
        //else { log_debug("I didn't resize the image ^_^\n"); }
#else // TRY_LETTER_BOX
        // strategy: resize and attach to letter box
        image resized = resize_image(im, im_rw, im_rh);
        // init letter box
        for (k = 0; k < lb.c; k++) {
            idx = lb.h * lb.w * k;
            for (i = 0; i < lb.h * lb.w; i++) {
                lb.data[idx + i] = (0. - voc_image_cov[0][k]) * voc_image_cov[1][k];
            }
        }
        // attach resized image to letter box
        add_image(resized, lb, lb_ow, lb_oh);
        // release resized image
        free_image(resized);
        // reset data buffer reference
        _data = lb.data;
#endif // !TRY_LETTER_BOX
        // nchw pre-process
        for (k = 0; k < lb.c; k++) {
            for (i = 0; i < lb.h; i++) {
                for (j = 0; j < lb.w; j++) {
                    idx = k * lb.h * lb.w + i * lb.w + j;
                    lb.data[idx] = (_data[idx] - voc_image_cov[0][k]) * voc_image_cov[1][k];
                }
            }
        }
    }
    else {
        get_input_data(image_file, lb.data, lb.h, lb.w, voc_image_cov[0], voc_image_cov[1]);
    }

    /* run graph */
    if (imi_utils_tm_run_graph(graph, repeat_count) < 0) {
        goto exit;
    }

    /* get output tensor shape: (1, x, 6, 1) */
    dim_num = get_tensor_shape(tensor, dims, DIM_NUM);
    /* Note: For mobilenet ssd caffe model, the output tensor shape isn't fixed.
        Everything is hard coded in the model, even the confidence and nms threshold.
        So we need to update output tensor shape here, which is different from yolo.
    log_echo("output tensor[%p] shape: %d(", tensor, dim_num);
    for (i = 0; i < dim_num; i++) {
        log_echo(" %d", dims[i]);
    }
    log_echo(")\n");
    */

    /* process the detection result */
    post_process_ssd(im, threshold, get_tensor_buffer(tensor), dims[1], target_class);

    // save result to output
    if (-1 != bgr) {
        imi_utils_image_save_permute_chw2hwc(fout, im, bgr);
        if (fc < frame) goto read_data;
    }
    else {
        save_image(im, output_file);
    }

exit:
    free_image(im);
    free_image(lb);
    if (fp) fclose(fp);
    if (fout) fclose(fout);
    if (1 < fc) log_echo("total frame: %d\n", fc);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    /* show test status */
    imi_utils_tm_run_status(NULL);
    return 0;
}

