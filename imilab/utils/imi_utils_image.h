// ============================================================
//              Imilab Utils: Image Process APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/26
// ============================================================

#ifndef __IMI_UTILS_IMAGE_H__
#define __IMI_UTILS_IMAGE_H__

/* imilab includes */
#include "imi_utils_elog.h" // for: log_xxxx
/* tengine includes */
#include "tengine_operations.h" // for: image

/* the data type of input image */
#define IMG_DT_FP32     0
#define IMG_DT_FP16     1
#define IMG_DT_INT8     2
#define IMG_DT_UINT8    3
#define IMG_DT_INT32    4
#define IMG_DT_INT16    5

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Q: How to verify the output r/g/b data with origin data?
// A: try "cat image_r.dat image_g.dat image_b.dat > image.dat",
//  and "diff image.dat image__.dat" .
static __inline void _imi_utils_check_channel_1by1(const image img) {
    // check inputs
    if (NULL == img.data) {
        log_error("input image data invalid: NULL\n");
        return;
    }

    FILE *fp_rgb[] = {
        fopen("image_r.dat", "wb"),
        fopen("image_g.dat", "wb"),
        fopen("image_b.dat", "wb"),
        fopen("image__.dat", "wb"),
    };
    unsigned char uc;
    int c, h, w, off_c, off_h;
    for (c = 0; c < 3; c++) {
        off_c = img.w * img.h * c;
        for (h = 0; h < img.h; h++) {
            off_h = off_c + h * img.w;
            for (w = 0; w < img.w; w++) {
                char uc = (unsigned char)(*(img.data + w + off_h));
                fwrite(&uc, sizeof(uc), 1, fp_rgb[c]);
                fwrite(&uc, sizeof(uc), 1, fp_rgb[3]);
            }
        }
        fclose(fp_rgb[c]);
    }
}

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

    int rc = -1, idx, h, w, c;
    // Note: Here we must use unsigned type!
    unsigned char uc;
    for (h = 0; h < im.h; h++) {
        for (w = 0; w < im.w; w++) {
            for (c = 0; c < channels; c++) {
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

// load bgr24/bgra32 as rgb planar to letter box without resize
// @param:  img[out] input raw image
// @param:  bgr[in]  input raw data format
// @param:  lb[out]  output letter box
// @param:  cov[in]  mean and scale
static int imi_utils_image_load_letterbox(
    FILE *fp, const image img, char bgr, const image lb, const float cov[][3]) {
    // todo: optimize/resize input image
    if ((img.w <= img.h && lb.h != img.h) ||
        (img.h <= img.w && lb.w != img.w)) {
        log_error("input size (%d, %d) not match letter box size (%d, %d)!\n", img.w, img.h, lb.w, lb.h);
        log_error("please try to resize the input image first!\n");
        exit(0);
    }

    // check inputs
    if (NULL == fp || NULL == img.data || NULL == lb.data) {
        log_error("invalid input param NULL\n");
        return -1;
    }

    int h, w, c, idx, idx_;
    // init letter box (nchw)
    for (c = 0; c < lb.c; c++) {
        idx_ = lb.h * lb.w * c;
        for (idx = 0; idx < lb.h * lb.w; idx++) {
            lb.data[idx_ + idx] = (0. - cov[0][c]) * cov[1][c];
        }
    }

    // Note: Here we must use unsigned type!
    unsigned char uc;
    int rc = -1, dw = (lb.w - img.w) / 2, dh = (lb.h - img.h) / 2;
    for (h = 0; h < img.h; h++) {
        for (w = 0; w < img.w; w++) {
            for (c = 0; c < img.c; c++) {
                rc = fread(&uc, sizeof(unsigned char), 1, fp);
                if (1 != rc) {
                    return feof(fp) ? 0 : -1;
                }
                if (c < lb.c) {
                    if (bgr) idx = w + img.w * h + (img.c - 1 - c) * img.w * img.h;
                    else idx = w + img.w * h + c * img.w * img.h;
                    //printf("w=%d h=%d c=%d idx=%d\n", w, h, c, idx);
                    img.data[idx] = (float)uc;
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

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !__IMI_UTILS_IMAGE_H__
