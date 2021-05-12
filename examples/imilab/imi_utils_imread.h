// ============================================================
//              Imilab Utils:   Image Read APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_IMREAD_H__
#define __IMI_IMREAD_H__

/* std c includes */
#include <stdio.h>  // for: printf
#include <stdlib.h> // for: calloc
#include <stdint.h> // for: uint8_t
/* tengine includes */
#include "tengine_operations.h" // for: image

// Q: How to verify the output r/g/b data with origin data?
// A: try "cat image_r.dat image_g.dat image_b.dat > image.dat",
//  and "diff image.dat image__.dat" .
static __inline void _imi_utils_check_channel_1by1(const image &img) {
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
// @param:  bgr[in]     input raw data format
// @param:  channel[in] input raw data channels
static int imi_utils_load_image(FILE *fp, image &img, char bgr, int channels) {
    int img_size = img.w * img.h * img.c;
    if (NULL == img.data) {
        img.data = (float *)calloc(sizeof(float), img_size);
    }

    int rc = -1, idx;
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
    //_imi_utils_check_channel_1by1(img);

    return rc;
}

// load bgr24/bgra32 as rgb planar to letter box without resize
// @param:  img[out] input raw image
// @param:  bgr[in]  input raw data format
// @param:  lb[out]  output letter box
// @param:  cov[in]  mean and scale
static int imi_utils_load_letterbox(FILE *fp, image &img, char bgr, image &lb, const float cov[][3]) {
    int img_size = img.w * img.h * img.c;
    if (NULL == img.data) {
        img.data = (float *)calloc(sizeof(float), img_size);
    }
    int lb_size = lb.w * lb.h * lb.c;
    if (NULL == lb.data) {
        lb.data = (float *)calloc(sizeof(float), lb_size);
    }

    int rc = -1, idx, idx_;
    // init letter box
    for (idx = 0; idx < lb_size; idx++) {
        lb.data[idx] = .5;
    }

    int dw = (lb.w - img.w) / 2, dh = (lb.h - img.h) / 2;
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
                    //printf("w=%d h=%d c=%d idx=%d\n", w, h, c, idx);
                    img.data[idx] = (float)b;
                    if (bgr) idx_ = (w + dw) + lb.w * (h + dh) + (img.c - 1 - c) * lb.w * lb.h;
                    else idx_ = (w + dw) + lb.w * (h + dh) + c * lb.w * lb.h;
                    lb.data[idx_] = (img.data[idx] - cov[0][c]) * cov[1][c];
                }
            }
        }
    }

    // check channel one by one(default: R G B)
    //_imi_utils_check_channel_1by1(img);
    return rc;
}

#endif // !__IMI_IMREAD_H__
