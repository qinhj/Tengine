## Quick Build ##
```
$ mkdir -p build_new && cd build_new
$ cmake .. -DTENGINE_BUILD_TESTS=ON
```

## Quick Test ##
```
$ export PATH=/home/qinhj/desktop/AI/tengine-lite/build_new/examples:$PATH
$ export MODEL=/media/sf_Workshop

$ tm_yolov3_tiny_ -m $MODEL/yolov3_tiny.tmfile -i $MODEL/imilab_640x360_catdog.bgr -o 640x360_catdog_0.rgb24 -f 1000
$ tm_yolov3_tiny_ -m $MODEL/yolov3_tiny.tmfile -i $MODEL/imilab_640x360_catdog.bgra -o 640x360_catdog_1.rgb24 -f 1000

$ tm_retinaface_ -m $MODEL/retinaface.tmfile -i $MODEL/imilab_640x360_human1.bgr -o 640x360_h1_0.rgb24 -f 1000
$ tm_retinaface_ -m $MODEL/retinaface.tmfile -i $MODEL/imilab_640x360_human1.bgra -o 640x360_h1_1.rgb24 -f 1000
$ tm_retinaface_ -m $MODEL/retinaface.tmfile -i $MODEL/imilab_640x360_human2.bgr -o 640x360_h2_0.rgb24 -f 1000
$ tm_retinaface_ -m $MODEL/retinaface.tmfile -i $MODEL/imilab_640x360_human2.bgra -o 640x360_h2_1.rgb24 -f 1000
$ tm_retinaface_ -m $MODEL/retinaface.tmfile -i $MODEL/imilab_960x512_human.bgr -w 960 -h 512 -o 960x512_h3_0.rgb24 -f 1000
$ tm_retinaface_ -m $MODEL/retinaface.tmfile -i $MODEL/imilab_960x512_human.bgra -w 960 -h 512 -o 960x512_h3_1.rgb24 -f 1000
```

## FAQ ##
```
1. How to increase face detection precision(e.g. tm_retinaface)?
A: try to increase confidence threshold, e.g.
- const float CONF_THRESH = 0.8f;
+ const float CONF_THRESH = 0.9f;
```
