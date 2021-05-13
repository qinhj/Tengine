// @Author: qinhongjie@imilab.com

#ifdef _MSC_VER
#define NOMINMAX
#endif

/* std c++ includes */
#include <vector>
/* tengine includes */
#include "common.h"
//#include "tengine/c_api.h"
#include "tengine_operations.h" // for: imread
/* imilab includes */
#include "imilab/imi_utils_imread.h"

#define IMAGE_PATH  "/media/sf_Workshop/color_375x375_rgb888.bmp"
#define OUTPUT_PATH "output.rgb"

struct Size2i {
    int width;
    int height;
};

static const char *image_file = IMAGE_PATH;
static const char *output_file = OUTPUT_PATH;
static Size2i image_size = { 640, 360 };
static int frame = 1;

static int get_input_data(const char* image_file, std::vector<float>& image_data, Size2i& size) {
    image img = imread(image_file);

    size.width = img.w;
    size.height = img.h;
    printf("%s w: %d, h: %d\n", image_file, img.w, img.h);

    // check whole image data(default: float32)
    FILE *fp = fopen("image.dat", "wb");
    fwrite(img.data, sizeof(float), img.w * img.h * img.c, fp);
    fclose(fp);
    // check channel one by one(default: R G B)
    _imi_utils_check_channel_1by1(img);

    int img_size = img.w * img.h * img.c;

#if 0
    img = image_premute(img);
    // check premuted image data(default: float32)
    FILE *fp_p = fopen("image_premute.dat", "wb");
    fwrite(img.data, sizeof(float), img.w * img.h * img.c, fp_p);
    fclose(fp_p);
#endif

    image_data.resize(img_size);

    memcpy(image_data.data(), img.data, img_size * sizeof(float));

    free_image(img);

    return img_size;
}

static void show_usage() {
    printf("[Usage]:  [-u]\n");
    printf("    [-i input_file] [-o output_file] [-w width] [-h height] [-f max_frame]\n");
    printf("[Example]:\n");
    printf("    ./examples/tm_test_imread -i /media/sf_Workshop/imilab_640x360x4_bgra_catdog.rgb -f 2\n");
}

static void test_load_data(int channel, char bgr) {
    FILE *fout = output_file ? fopen(output_file, "wb") : NULL;
    FILE *fin = fopen(image_file, "rb");

    image img = make_image(image_size.width, image_size.height, 3);
    int img_size = img.w * img.h * img.c, rc = 0;
    unsigned char uc;
    while (rc < frame) {
        if (imi_utils_load_image(fin, img, bgr, channel) != 1) {
            printf("get_input_data error!\n");
            break;
        }
        else {
            if (fout) {
                for (int i = 0; i < img_size; i++) {
                    uc = (unsigned char)(*(img.data + i));
                    fwrite(&uc, sizeof(unsigned char), 1, fout);
                }
            }
            rc += 1;
            // check channel one by one(default: R G B)
            _imi_utils_check_channel_1by1(img);
        }
    }

    fclose(fin);
    if (fout) fclose(fout);
    free(img.data);
    printf("total frame: %d\n", rc);
}


int main(int argc, char* argv[]) {

    int res;
    while ((res = getopt(argc, argv, "i:o:w:h:f:u")) != -1) {
        switch (res) {
        case 'i':
            image_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'w':
            image_size.width = atoi(optarg);
            break;
        case 'h':
            image_size.height = atoi(optarg);
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

    if (image_file == nullptr) {
        printf("Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(image_file))
        return -1;

    if (strstr(image_file, "bgra")) {
        test_load_data(4, 1);
    }
    else if (strstr(image_file, "bgr")) {
        test_load_data(3, 1);
    }
    else if (strstr(image_file, "rgba")) {
        test_load_data(4, 0);
    }
    else if (strstr(image_file, "rgb")) {
        test_load_data(3, 0);
    }
    else {
        std::vector<float> image_data;
        int img_size = get_input_data(image_file, image_data, image_size);
#if 0
        // check premuted image data(default: float32)
        FILE *fp_ = fopen("image_final.dat", "wb");
        fwrite(image_data.data(), sizeof(float), img_size, fp_);
        fclose(fp_);
#endif
    }

    return 0;
}
