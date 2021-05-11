// ============================================================
//                  Imilab Utils: Image IO APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/04/28
// ============================================================

#ifndef __IMI_IMREAD_H__
#define __IMI_IMREAD_H__

/* std c includes */
#include <stdio.h>  // for: printf
#include <stdlib.h> // for: calloc
/* std c++ includes */
#include <cmath>    // for: round
/* tengine includes */
#include "tengine_operations.h" // for: image

// Q: How to verify the output r/g/b data with origin data?
// A: try "cat image_r.dat image_g.dat image_b.dat > image.dat",
//  and "diff image.dat image__.dat" .
static void _check_channel_1by1(const image &img) {
    FILE *fp_rgb[] = {
        fopen("image_r.dat", "wb"),
        fopen("image_g.dat", "wb"),
        fopen("image_b.dat", "wb"),
        fopen("image__.dat", "wb"),
    };
    for (int c = 0; c < 3; c++) {
        int off_c = img.w * img.h * c;
        for (int h = 0; h < img.h; h++) {
            int off_h = off_c + h * img.w;
            for (int w = 0; w < img.w; w++) {
                unsigned char uc = (unsigned char)(*(img.data + w + off_h));
                fwrite(&uc, sizeof(uc), 1, fp_rgb[c]);
                fwrite(&uc, sizeof(uc), 1, fp_rgb[3]);
            }
        }
        fclose(fp_rgb[c]);
    }
}

// load bgr24/bgra32 as rgb planar
// @param:  fp[in/out]  input file pointer
// @param:  img[in/out] target output image obj
// @param:  channel[in] input raw data channels
// @param:  bgr[in]     input raw data format
static int get_input_data(FILE *fp, image &img, int channels, char bgr) {
    int img_size = img.w * img.h * img.c, rc = -1, idx;
    if (NULL == img.data) {
        img.data = (float *)calloc(sizeof(float), img_size);
    }
    // Note: Here we must use unsigned type!
    unsigned char b;
    for (int h = 0; h < img.h; h++) {
        for (int w = 0; w < img.w; w++) {
            for (int c = 0; c < channels; c++) {
                rc = fread(&b, sizeof(unsigned char), 1, fp);
                if (1 != rc) {
                    return feof(fp) ? 0 : -1;
                }
                if (c < img.c) {
                    if (bgr) idx = w + img.w * h + (img.c - 1 - c) * img.w * img.h;
                    else idx = w + img.w * h + c * img.w * img.h;
                    //printf("%d %d %d: %d\n", w, h, c, idx);
                    img.data[idx] = (float)b;
                }
            }
        }
    }
    // check channel one by one(default: R G B)
    //_check_channel_1by1(img);

    return rc;
}

static int get_input_data_focus_yolov5(const float *data, image &lb) {
    /* focus process: 3x640x640 -> 12x320x320 */
    /*
     | 0 2 |          C0-0, C1-0, C2-0,
     | 1 3 | x C3 =>  C0-1, C1-1, C2-1, x C12
                      C0-2, C1-2, C2-2,
                      C0-3, C1-3, C2-3,
    */
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
static int get_input_data_focus_yolov5_uint8(const float *data, image &lb, float input_scale, int zero_point) {
    /* focus process: 3x640x640 -> 12x320x320 */
    /*
     | 0 2 |          C0-0, C1-0, C2-0,
     | 1 3 | x C3 =>  C0-1, C1-1, C2-1, x C12
                      C0-2, C1-2, C2-2,
                      C0-3, C1-3, C2-3,
    */
    uint8_t *input_data = (uint8_t *)(lb.data);
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

                        /* quant to uint8 */
                        int udata = (int)round(data[in_index] / input_scale + (float)zero_point);
                        if (255 < udata) udata = 255;
                        else if (udata < 0) udata = 0;
                        input_data[out_index] = udata;
                    }
                }
            }
        }
    }
    return 0;
}

// load bgr24/bgra32 as rgb planar to letter box without resize
// @param:  lb[out]  output letter box
// @param:  bgr[in]  input raw data format
// @param:  img[out] input raw image
// @param:  cov[in]  mean and scale
static int get_input_data_yolov5(FILE *fp, image &lb, char bgr, image &img, const float cov[][3]) {
    int lb_size = lb.w * lb.h * lb.c;
    if (NULL == lb.data) {
        lb.data = (float *)calloc(sizeof(float), lb_size);
    }

    if (640 != lb.w || 640 != lb.h) {
        fprintf(stderr, "[%s] yolov5 letter box size must be: 640x640!\n", __FUNCTION__);
        exit(0);
    }
    // todo: optimize/resize input image
    if ((img.w <= img.h && 640 != img.h) ||
        (img.h <= img.w && 640 != img.w)) {
        fprintf(stderr, "[%s] input size (%d, %d) not match letter box size (%d, %d)!\n", __FUNCTION__, img.w, img.h, lb.w, lb.h);
        fprintf(stderr, "[%s] please try to resize the input image first!\n", __FUNCTION__);
        exit(0);
    }

    static float *data = (float *)calloc(sizeof(float), lb_size);
    //printf("mean:  %.3f, %.3f, %.3f\n", cov[0][0], cov[0][1], cov[0][2]);
    //printf("scale: %.3f, %.3f, %.3f\n", cov[1][0], cov[1][1], cov[1][2]);


    int idx;
    // init letter box
    for (idx = 0; idx < lb_size; idx++) {
        /*lb.*/data[idx] = .5;
    }

    int rc = -1, idx_, dw = (lb.w - img.w) / 2, dh = (lb.h - img.h) / 2;
    // Note: Here we must use unsigned type!
    unsigned char b;
    for (int h = 0; h < img.h; h++) {
        for (int w = 0; w < img.w; w++) {
            for (int c = 0; c < img.c; c++) {
                rc = fread(&b, sizeof(unsigned char), 1, fp);
                if (1 != rc) {
                    return feof(fp) ? 0 : -1;
                }
                if (c < lb.c) {
                    if (bgr) idx = w + img.w * h + (img.c - 1 - c) * img.w * img.h;
                    else idx = w + img.w * h + c * img.w * img.h;
                    img.data[idx] = (float)b;
                    if (bgr) idx_ = (w + dw) + lb.w * (h + dh) + (img.c - 1 - c) * lb.w * lb.h;
                    else idx_ = (w + dw) + lb.w * (h + dh) + c * lb.w * lb.h;
                    //printf("w=%d h=%d c=%d idx=%d\n", w, h, c, idx);
                    /*lb.*/data[idx_] = (img.data[idx] - cov[0][c]) * cov[1][c];
                }
            }
        }
    }
    // check channel one by one(default: R G B)
    //_check_channel_1by1(img);

    // todo: optimize
    get_input_data_focus_yolov5(data, lb);
    return rc;
}
static int get_input_data_yolov5_uint8(FILE *fp, image &lb, char bgr, image &img, const float cov[][3], float input_scale, int zero_point) {
    int lb_size = lb.w * lb.h * lb.c;
    if (NULL == lb.data) {
        lb.data = (float *)calloc(sizeof(float), lb_size);
    }

    if (640 != lb.w || 640 != lb.h) {
        fprintf(stderr, "[%s] yolov5 letter box size must be: 640x640!\n", __FUNCTION__);
        exit(0);
    }
    // todo: optimize/resize input image
    if ((img.w <= img.h && 640 != img.h) ||
        (img.h <= img.w && 640 != img.w)) {
        fprintf(stderr, "[%s] input size (%d, %d) not match letter box size (%d, %d)!\n", __FUNCTION__, img.w, img.h, lb.w, lb.h);
        fprintf(stderr, "[%s] please try to resize the input image first!\n", __FUNCTION__);
        exit(0);
    }

    static float *data = (float *)calloc(sizeof(float), lb_size);
    //printf("mean:  %.3f, %.3f, %.3f\n", cov[0][0], cov[0][1], cov[0][2]);
    //printf("scale: %.3f, %.3f, %.3f\n", cov[1][0], cov[1][1], cov[1][2]);


    int idx;
    // init letter box
    for (idx = 0; idx < lb_size; idx++) {
        /*lb.*/data[idx] = .5;
    }

    int rc = -1, idx_, dw = (lb.w - img.w) / 2, dh = (lb.h - img.h) / 2;
    // Note: Here we must use unsigned type!
    unsigned char b;
    for (int h = 0; h < img.h; h++) {
        for (int w = 0; w < img.w; w++) {
            for (int c = 0; c < img.c; c++) {
                rc = fread(&b, sizeof(unsigned char), 1, fp);
                if (1 != rc) {
                    return feof(fp) ? 0 : -1;
                }
                if (c < lb.c) {
                    if (bgr) idx = w + img.w * h + (img.c - 1 - c) * img.w * img.h;
                    else idx = w + img.w * h + c * img.w * img.h;
                    img.data[idx] = (float)b;
                    if (bgr) idx_ = (w + dw) + lb.w * (h + dh) + (img.c - 1 - c) * lb.w * lb.h;
                    else idx_ = (w + dw) + lb.w * (h + dh) + c * lb.w * lb.h;
                    //printf("w=%d h=%d c=%d idx=%d\n", w, h, c, idx);
                    /*lb.*/data[idx_] = (img.data[idx] - cov[0][c]) * cov[1][c];
                }
            }
        }
    }
    // check channel one by one(default: R G B)
    //_check_channel_1by1(img);

    // todo: optimize
    get_input_data_focus_yolov5_uint8(data, lb, input_scale, zero_point);
    return rc;
}

#endif // !__IMI_IMREAD_H__
