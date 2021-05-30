#!/bin/bash

set -e # -xe

## env settings
export ROOTDIR=$(pwd)
export LD_LIBRARY_PATH=${ROOTDIR}/install/lib:$LD_LIBRARY_PATH
export PATH=${ROOTDIR}/install/bin:$PATH
export MODELS=${ROOTDIR}/models/320x320/
export IMAGES=${ROOTDIR}/datasets/320x180/
export OUTPUT=${ROOTDIR}/outputs/320x320/

## settings
MD5SUM=9511de587a710e3cb28b4de63894c831

i=1
MD5=$MD5SUM
while [ x"$MD5" = x"$MD5SUM" ] ; do
    echo "test $i"
    i=$(($i+1))
    rm -rf imilab_320x180x3_bgr_human3.*
    sleep 1
    tm_yolov5s_ -m $MODELS/yolov5s.imi.v3.opt.tmfile -i $IMAGES/imilab_320x180x3_bgr_human3.rgb -o imilab_320x180x3_bgr_human3.rgb -t4 -n1 -f200 -w320 -h180 > imilab_320x180x3_bgr_human3.txt
    MD5=$(md5sum imilab_320x180x3_bgr_human3.rgb)
    MD5=${MD5%% *}
done
