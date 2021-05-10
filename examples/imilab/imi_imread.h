// ============================================================
//                  Imilab Utils: Image IO APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/04/28
// ============================================================

#ifndef __IMI_IMREAD_H__
#define __IMI_IMREAD_H__

#include <stdio.h>  // for: printf
#include <stdlib.h> // for: calloc
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

#endif // !__IMI_IMREAD_H__
