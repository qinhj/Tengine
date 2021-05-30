#!/bin/sh

set -e # -xe

## env settings
export ROOTDIR=$(pwd)
export LD_LIBRARY_PATH=${ROOTDIR}/install/lib:$LD_LIBRARY_PATH
export PATH=${ROOTDIR}/install/bin:$PATH
export MODELS=${ROOTDIR}/models/320x320/
export IMAGES=${ROOTDIR}/datasets/320x180/
export OUTPUT=${ROOTDIR}/outputs/320x320/

rm -rf $OUTPUT
mkdir -p $OUTPUT

## official models
MDL_YOLOV3_TINY=yolov3-tiny.v9.5.opt.tmfile
MDL_YOLOV5S=yolov5s.v5.opt.tmfile

check_models_official() {
    ## yolov3-tiny(coco: 80 classes, draw person)
    mkdir -p $OUTPUT/yolov3_tiny
    for img in $(ls $IMAGES); do tm_yolov3_tiny_ -m $MODELS/$MDL_YOLOV3_TINY -i $IMAGES/$img -o $OUTPUT/yolov3_tiny/$img -t 4 -c 0 -f 500 -w 320 -h 180; done

    ## yolov5s(coco: 80 classes, draw person)
    mkdir -p $OUTPUT/yolov5s
    for img in $(ls $IMAGES); do tm_yolov5s_ -m $MODELS/$MDL_YOLOV5S -i $IMAGES/$img -o $OUTPUT/yolov5s/$img -t 4 -c 0 -f 500 -w 320 -h 180; done
}

## imilab models
IMI_YOLOV3_TINY=yolov3-tiny.imi.v1.opt.tmfile
IMI_YOLOV5S=yolov5s.imi.v3.opt.tmfile

check_models_imilab() {
    ## yolov3-tiny(coco: person)
    mkdir -p $OUTPUT/yolov3_tiny.imi
    for img in $(ls $IMAGES); do tm_yolov3_tiny_ -m $MODELS/$IMI_YOLOV3_TINY -i $IMAGES/$img -o $OUTPUT/yolov3_tiny.imi/$img -t 4 -n 1 -f 500 -w 320 -h 180; done

    ## yolov5s(coco: person)
    mkdir -p $OUTPUT/yolov5s.imi
    for img in $(ls $IMAGES); do tm_yolov5s_ -m $MODELS/$IMI_YOLOV5S -i $IMAGES/$img -o $OUTPUT/yolov5s.imi/$img -t 4 -n 1 -f 500 -w 320 -h 180; done
}

check_models_official
check_models_imilab
