#include <stdio.h>
#include <stdlib.h>

typedef struct image
{
    int w;
    int h;
    int c;
    float* data;
} image;

int main(int argc, char* argv[])
{
    image img = {640, 360, 3, NULL};
    img.data = (float*)calloc(sizeof(float), img.w * img.h * img.c);

    char* file = argv[1];
    FILE* fp = fopen(file, "rb");
    fread(img.data, sizeof(float), img.w * img.h * img.c, fp);
    fclose(fp);

    FILE* fp_rgb[] = {
        fopen("image_r.dat", "wb"),
        fopen("image_g.dat", "wb"),
        fopen("image_b.dat", "wb"),
        fopen("image__.dat", "wb"),
    };
    int c, h, w, off_c, off_h;
    unsigned char uc;
    for (c = 0; c < 3; c++)
    {
        off_c = img.w * img.h * c;
        for (h = 0; h < img.h; h++)
        {
            off_h = off_c + h * img.w;
            for (w = 0; w < img.w; w++)
            {
                uc = (unsigned char)(*(img.data + w + off_h));
                fwrite(&uc, sizeof(uc), 1, fp_rgb[c]);
                fwrite(&uc, sizeof(uc), 1, fp_rgb[3]);
            }
        }
        fclose(fp_rgb[c]);
    }

    free(img.data);
    return 0;
}
