#!/bin/sh

export ROOTDIR=$(pwd)
export LD_LIBRARY_PATH=${ROOTDIR}/install/lib:$LD_LIBRARY_PATH
export PATH=${ROOTDIR}/install/bin:$PATH
export IMAGES=/media/sf_Workshop/Dataset/raw/images/
export OUTPUT=${ROOTDIR}/output
export OUTNAME=imilab_640x360x3_human1-1

## retinaface
tm_retinaface -m retinaface.tmfile -i $IMAGES/imilab_640x360x3_human1-1.bmp && mv tengine_example_out.jpg ${OUTPUT}/${OUTNAME}_retinaface.jpg
tm_retinaface_ -m retinaface.tmfile -i $IMAGES/imilab_640x360x3_bgr_human1-1.rgb24 -o ${OUTPUT}/${OUTNAME}_retinaface.rgb24

## yolov5s(coco: person)
tm_yolov5s_ -m yolov5s.tmfile -i $IMAGES/imilab_640x360x3_bgr_human1-1.rgb24 -o ${OUTPUT}/human1-1_yolov5s.rgb24 -n 1
tm_yolov5s_uint8_ -m yolov5s.uint8.tmfile -i $IMAGES/imilab_640x360x3_bgr_human1-1.rgb24 -o ${OUTPUT}/${OUTNAME}_yolov5s.uint8.rgb24 -n 1

## yolov5s(coco: 80 classes)
tm_yolov5s -m yolov5s.v5.tmfile -i $IMAGES/imilab_640x360x3_human1-1.bmp && mv yolov5_out.jpg ${OUTPUT}/${OUTNAME}_yolov5s.v5.jpg
tm_yolov5s_ -m yolov5s.v5.tmfile -i $IMAGES/imilab_640x360x3_bgr_human1-1.rgb24 -o ${OUTPUT}/${OUTNAME}_yolov5s.v5.rgb24
tm_yolov5s_uint8 -m yolov5s.v5.uint8.tmfile -i $IMAGES/imilab_640x360x3_human1-1.bmp && mv yolov5s_uint8_out.jpg ${OUTPUT}/${OUTNAME}_yolov5s.v5.uint8.jpg
tm_yolov5s_uint8_ -m yolov5s.v5.uint8.tmfile -i $IMAGES/imilab_640x360x3_bgr_human1-1.rgb24 -o ${OUTPUT}/${OUTNAME}_yolov5s.v5.uint8.rgb24
