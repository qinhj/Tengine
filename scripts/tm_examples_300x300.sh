#!/bin/sh

set -e # -xe

## env settings
export ROOTDIR=$(pwd)
export LD_LIBRARY_PATH=${ROOTDIR}/install/lib:$LD_LIBRARY_PATH
export PATH=${ROOTDIR}/install/bin:$PATH
export MODELS=${ROOTDIR}/models/
export IMAGES=${ROOTDIR}/datasets/300x300/
export OUTPUT=${ROOTDIR}/outputs/300x300/

rm -rf $OUTPUT
mkdir -p $OUTPUT

## official models
MDL_MOBILENET_SSD=mobilenet_ssd.tmfile

check_models_official() {
    ## mobilenet ssd(voc: 21 classes)
    for img in $(ls $IMAGES); do tm_mobilenet_ssd_ -m $MODELS/$MDL_MOBILENET_SSD -i $IMAGES/$img -o $OUTPUT/$img -t 4 -c 15 -f 500 -h 300 -w 300; done
}

check_models_official
