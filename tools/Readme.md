## Tengine-Convert-Tools ##
[Convert Toolkit](https://github.com/OAID/Tengine-Convert-Tools.git)  

## Convert Toolkit: Build ##
```
## check protoc version
$ protoc --version
libprotoc 2.6.1
## update to latest version
$ sudo add-apt-repository ppa:maarten-fonville/protobuf
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get dist-upgrade
## remove old ones
$ sudo apt-get purge libprotobuf-dev protobuf-compiler protobuf-c-compiler
$ dpkg -l | grep protobuf

## install dependencies
$ sudo apt install libprotobuf-dev protobuf-compiler

## build convert tool
$ mkdir build && cd build
$ cmake ..
$ make -j`nproc` && make install
```

## Convert Toolkit: Run ##
```
## how to use
$ ./build/install/bin/convert_tool -h
[Convert Tools Info]: optional arguments:
	-h    help            show this help message and exit
	-f    input type      path to input float32 tmfile
	-p    input structure path to the network structure of input model(*.prototxt, *.symbol, *.cfg, *.pdmodel)
	-m    input params    path to the network params of input model(*.caffemodel, *.params, *.weight, *.pb, *.onnx, *.tflite, *.pdiparams)
	-o    output model    path to output fp32 tmfile

[Convert Tools Info]: example arguments:
	./convert_tool -f caffe -p ./mobilenet.prototxt -m ./mobilenet.caffemodel -o ./mobilenet.tmfile

## examples(onnx)
$ ./build/install/bin/convert_tool -f onnx -m mobilenet.onnx -o mobilenet.tmfile
$ ./build/install/bin/convert_tool -f darknet -p yolov3.cfg -m yolov3.weights -o yolov3.tmfile

## examples(yolov3: pytorch => onnx => tmfile)
$ python3 models/export.py --weights weights/yolov3.v9.5.pt --simplify
$ python3 yolov3-opt.py --input yolov3.onnx --output yolov3-opt.onnx --cut "Sigmoid_189,Sigmoid_238,Sigmoid_287"
$ python3 yolov3-opt.py --input yolov3-spp.onnx --output yolov3-spp-opt.onnx --cut "Sigmoid_195,Sigmoid_244,Sigmoid_293"
$ python3 yolov3-opt.py --input yolov3-tiny.onnx --output yolov3-tiny-opt.onnx --cut "Sigmoid_62,Sigmoid_111"

## examples(yolov5s: pytorch => onnx => tmfile)
$ python3 models/export.py --weights weights/yolov5s.v5.pt --simplify
$ cd tools/optimize
$ python3 yolov5s-opt.py --input yolov5s.v4.onnx --output yolov5s.v4.opt.onnx --in_tensor 167 --out_tensor 381,420,459
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s.v5.opt.onnx --in_tensor 167 --out_tensor 397,458,519
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s-p3p4.opt.onnx --in_tensor 167 --out_tensor 397,458
$ python3 yolov5s-opt.py --input yolov5s-tiny.onnx --output yolov5s-tiny.opt.onnx --in_tensor 121 --out_tensor 272,333
$ ./install/bin/convert_tool -f onnx -m yolov5/yolov5s.opt.onnx -o yolov5/yolov5s.opt.tmfile

## Note:
1) see https://github.com/OAID/Tengine-Convert-Tools for more details;
```

## Quick Quant (Ubuntu18.04) ##
[quant_tool_int8](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_int8)  
```
## dependent libraries(opencv-3.2.0.zip)
$ sudo apt install libopencv-dev

## do post training quantization
$ MODELS=/media/sf_Workshop/Models/coco_nc80
$ IMAGES=/media/sf_Workshop/Dataset/Yolo3/calibration/
$ ./quant_tool_uint8 -m $MODELS/yolov5s.v5.tmfile -i $IMAGES -o yolov5s_uint8.tmfile -g 12,320,320 -w 0,0,0 -s 0.004,0.004,0.004 -c 0 -k 1 -y 640,640 -t 2
## Todo: quant retinaface model
$ ./quant_tool_uint8 -m $MODELS/retinaface.tmfile -i $IMAGES -o retinaface.uint8.tmfile -w 0,0,0 -t 2

## Note:
0) the image num of calibration dataset we suggest to use 500-1000;
1) during test, one'd better use "-t 2" then "-t 1";
2) options:
-g  size            the size of input image(using the resize the original image, default is 3,224,224)
-w  mean            value of mean (mean value, default is 104.0,117.0,123.0)
-s  scale           value of normalize (scale value, default is 1.0,1.0,1.0)
-t  num thread      count of processing threads(default is 1)
-c  center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
-y  letter box      flag which indicates that letter box process image is necessary(maybe using for YOLOv3/v4, 0:OFF, 1:ON, default is 0)
-k  focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)

## smoke test
$ for img in $(ls $IMAGES/); do tm_yolov5s_uint8 -m yolov5s_uint8.tmfile -i $IMAGES/$img -r 1 -t 1 && mv yolov5s_uint8_out.jpg output/${img%.*}.jpg; done
## Todo: test quantized retinaface
$ for img in $(ls $IMAGES/); do tm_retinaface_uint8 -m retinaface_uint8.tmfile -i $IMAGES/$img && mv retinaface_uint8_out.jpg output/${img%.*}.jpg; done
```

## Others ##
```
* split_dat.c
Split float32 image data into R/G/B and U8C3 data.
```

## Quick FAQ ##
```
1. How does multithread work?
A: Try OpenBLAS and see the following issues for more details:
https://github.com/OAID/Tengine/issues/49
https://github.com/OAID/Tengine/issues/137

2. Convert Error: {ConstantOfShape} not supported for yolov3
A: add "--simplify" when export yolov3 or yolov5 models.
https://github.com/OAID/Tengine-Convert-Tools/issues/39

3. /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.27' not found
A: run "strings /lib/x86_64-linux-gnu/libm.so.6 | grep GLIBC_" and try ubuntu18.04
```
