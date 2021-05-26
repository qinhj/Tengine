// ============================================================
//              Imilab Utils: Image Process APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/26
// ============================================================

#ifndef __IMI_UTILS_IMAGE_H__
#define __IMI_UTILS_IMAGE_H__

/* std c includes */
#include <stdio.h>  // for: FILE
/* tengine includes */
#include "tengine_operations.h" // for: image

/* the data type of input image */
#define IMG_DT_FP32     0
#define IMG_DT_FP16     1
#define IMG_DT_INT8     2
#define IMG_DT_UINT8    3
#define IMG_DT_INT32    4
#define IMG_DT_INT16    5

// @brief:  load image raw data as it is
// @param:  im[out] image info
// @param:  file[in] input file name
// @param:  dt[in] input data type
static int imi_utils_image_load_raw(const image im, const char *file, int dt) {
    // check inputs
    if (NULL == file || NULL == im.data) {
        return -1;
    }

    int img_size = im.w * im.h * im.c, rc = 0;
    FILE *fin = fopen(file, "rb");
    if (fin) {
        switch (dt) {
            case IMG_DT_FP32:
            {
                rc = fread(im.data, sizeof(float), img_size, fin);
                break;
            }
            default: // as IMG_DT_UINT8
            {
                int i;
                unsigned char uc;
                for (i = 0; i < img_size; i++) {
                    rc += fread(&uc, sizeof(unsigned char), 1, fin);
                    im.data[i] = (float)uc;
                }
                break;
            }
        }
        fclose(fin);
    }
    return !(rc == img_size);
}

// @brief:  save image raw data as it is
// @param:  im[in] image info
// @param:  file[in] output file name
// @param:  dt[in] output data type
static int imi_utils_image_save_raw(const image im, const char *file, int dt) {
    // check inputs
    if (NULL == file || NULL == im.data) {
        return -1;
    }

    int img_size = im.w * im.h * im.c, rc = 0;
    FILE *fout = fopen(file, "wb");
    if (fout) {
        switch (dt) {
            case IMG_DT_FP32:
            {
                rc = fwrite(im.data, sizeof(float), img_size, fout);
                break;
            }
            default: // as IMG_DT_UINT8
            {
                int i;
                unsigned char uc;
                for (i = 0; i < img_size; i++) {
                    uc = (unsigned char)im.data[i];
                    rc += fwrite(&uc, sizeof(unsigned char), 1, fout);
                }
                break;
            }
        }
        fclose(fout);
    }
    return !(rc == img_size);
}

// @brief:  load image data from bgr24/bgra32(hwc) to bgr24/rgb24 planar format(chw)
// @param:  fp[in/out]  input file pointer
// @param:  img[in/out] output image instance
// @param:  bgr[in]     output channel order
// @param:  channel[in] input image channels
static int imi_utils_image_load_bgr(FILE *fp, const image im, char bgr, int channels) {
    // check inputs
    if (NULL == fp || NULL == im.data) {
        return -1;
    }

    int rc = -1, idx;
    // Note: Here we must use unsigned type!
    unsigned char uc;
    for (int h = 0; h < im.h; h++) {
        for (int w = 0; w < im.w; w++) {
            for (int c = 0; c < channels; c++) {
                rc = fread(&uc, sizeof(unsigned char), 1, fp);
                if (1 != rc) {
                    return feof(fp) ? 0 : -1;
                }
                if (c < im.c) {
                    if (bgr) idx = w + im.w * h + (im.c - 1 - c) * im.w * im.h;
                    else idx = w + im.w * h + c * im.w * im.h;
                    //printf("%d %d %d: %d\n", w, h, c, idx);
                    im.data[idx] = (float)uc;
                }
            }
        }
    }
    // check channel one by one(default: R G B)
    //_imi_utils_check_channel_1by1(img);

    return rc;
}

// @brief:  save image data from rgb planar to uint8 bgr format
// @param:  fp[in]  file pointer
// @param:  im[in]  image instance
// @param:  cs[in]  swap channel(RGB <-> BGR) or not
static int imi_utils_image_save_permute_chw2hwc(FILE *fp, const image im, char cs) {
    // check inputs
    if (NULL == fp || NULL == im.data || im.c < 3) {
        return -1;
    }

    int img_size = im.w * im.h, rc = 0;
    int idx_0 = cs ? 2 : 0, idx_1 = 1, idx_2 = cs ? 0 : 2;
    if (fp) {
        int i;
        unsigned char uc[3];
        for (i = 0; i < img_size; i++) {
            uc[0] = (unsigned char)(*(im.data + i + idx_0 * img_size)); // b
            uc[1] = (unsigned char)(*(im.data + i + idx_1 * img_size)); // g
            uc[2] = (unsigned char)(*(im.data + i + idx_2 * img_size)); // r
            rc += fwrite(uc, 3 * sizeof(unsigned char), 1, fp);
        }
    }
    return !(rc == img_size);
}

#endif // !__IMI_UTILS_IMAGE_H__
