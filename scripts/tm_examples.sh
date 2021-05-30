#!/bin/sh

set -e # -xe

## env settings
export ROOTDIR=$(pwd)
export LD_LIBRARY_PATH=${ROOTDIR}/install/lib:$LD_LIBRARY_PATH
export PATH=${ROOTDIR}/install/bin:$PATH
export IMAGES=/media/sf_Workshop/Dataset/raw/images/
export OUTPUT=${ROOTDIR}/output

rm -rf $OUTPUT
mkdir -p $OUTPUT

## test image settings
IMG_BMP=imilab_640x360x3_human.bmp
IMG_BGR=imilab_640x360x3_human.bgr
IMG_OUT=imilab_640x360x3_human

## official models
MDL_RETINAFACE=retinaface.tmfile
MDL_MOBILENET_SSD=mobilenet_ssd.tmfile
MDL_YOLOV3=yolov3.opt.tmfile
MDL_YOLOV3_P4P5=yolov3-p4p5.opt.tmfile
MDL_YOLOV3_TINY=yolov3-tiny.opt.tmfile
MDL_YOLOV5S=yolov5s.v5.opt.tmfile
MDL_YOLOV5S_P3P4=yolov5s-p3p4.v5.opt.tmfile

check_examples() {
    ## retinaface
    tm_retinaface -m $MDL_RETINAFACE -i $IMAGES/$IMG_BMP && mv tengine_example_out.jpg ${OUTPUT}/${IMG_OUT}_retinaface.jpg

    ## mobilenet ssd(voc: 21 classes)
    tm_mobilenet_ssd -m $MDL_MOBILENET_SSD -i $IMAGES/$IMG_BMP && mv tengine_example_out.jpg ${OUTPUT}/${IMG_OUT}_mobilenet_ssd.jpg

    ## yolov5s(coco: 80 classes)
    tm_yolov5s -m $MDL_YOLOV5S -i $IMAGES/$IMG_BMP && mv yolov5_out.jpg ${OUTPUT}/${IMG_OUT}_yolov5s.jpg
}

MDL_YOLOV5S_UINT8=yolov5s.v5.opt.uint8.v2_800.tmfile
check_models_official() {
    ## retinaface
    tm_retinaface_ -m $MDL_RETINAFACE -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_retinaface.rgb24

    ## mobilenet ssd(voc: 21 classes)
    tm_mobilenet_ssd_ -m $MDL_MOBILENET_SSD -i $IMAGES/$IMG_BMP -o ${OUTPUT}/${IMG_OUT}_mobilenet_ssd
    tm_mobilenet_ssd_ -m $MDL_MOBILENET_SSD -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_mobilenet_ssd.rgb24

    ## yolov3(coco: 80 classes, draw person)
    tm_yolov3_ -m $MDL_YOLOV3 -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov3.rgb24 -c 0
    tm_yolov3_p4p5_ -m $MDL_YOLOV3_P4P5 -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov3_p4p5.rgb24 -c 0
    tm_yolov3_tiny_ -m $MDL_YOLOV3_TINY -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov3_tiny.rgb24 -c 0

    ## yolov5s(coco: 80 classes, draw person)
    tm_yolov5s_ -m $MDL_YOLOV5S -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov5s.rgb24 -c 0
    tm_yolov5s_p3p4_ -m $MDL_YOLOV5S_P3P4 -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov5s_p3p4.rgb24 -c 0

    ## yolov5s uint8(coco: 80 classes)
    tm_yolov5s_uint8 -m $MDL_YOLOV5S_UINT8 -i $IMAGES/$IMG_BMP && mv yolov5s_uint8_out.jpg ${OUTPUT}/${IMG_OUT}_yolov5s_uint8.jpg
}

## imilab models
IMI_YOLOV3_TINY=yolov3-tiny.imi.v1.opt.tmfile
IMI_YOLOV5S=yolov5s.opt.imi.v3.tmfile
IMI_YOLOV5S_TINY=yolov5s-tiny.opt.tmfile
IMI_YOLOV5S_P3P4=yolov5s-p3p4.opt.imi.v3.tmfile
IMI_YOLOV5S_UINT8=yolov5s.opt.imi.v1.uint8.765.tmfile

check_models_imilab() {
    ## yolov3 tiny(coco: person)
    tm_yolov3_tiny_ -m $IMI_YOLOV3_TINY -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov3_tiny.imi.rgb24 -n 1

    ## yolov5s(coco: person)
    tm_yolov5s_ -m $IMI_YOLOV5S -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov5s.imi.rgb24 -n 1
    tm_yolov5s_tiny_ -m $IMI_YOLOV5S_TINY -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov5s_tiny.imi.rgb24 -n 1
    tm_yolov5s_p3p4_ -m $IMI_YOLOV5S_P3P4 -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov5s_p3p4.imi.rgb24 -n 1

    ## yolov5s uint8(coco: person)
    tm_yolov5s_uint8_ -m $IMI_YOLOV5S_UINT8 -i $IMAGES/$IMG_BGR -o ${OUTPUT}/${IMG_OUT}_yolov5s_uint8.imi.rgb24 -n 1
}

check_examples
check_models_official
check_models_imilab
