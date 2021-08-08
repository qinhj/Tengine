# Tengine Lite #
https://github.com/OAID/Tengine  

## Quick Build ##
```
* requirements: g++, make, cmake(>=3.10)
$ sudo apt-get install build-essential
$ ...

## quick build
$ mkdir -p build && cd build
$ cmake .. -DTENGINE_BUILD_TESTS=ON -DTENGINE_FORCE_SKIP_OPENMP=OFF -DCMAKE_BUILD_TYPE=Debug
$ make && make install

* Note:
0) To build with static library, try:
-    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME})
+    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME}-static)
1) To build with arm-linux-gnueabi.toolchain.cmake
SET(TENGINE_TOOLCHAIN_FLAG "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4")
SET(TENGINE_FORCE_SKIP_OPENMP OFF)
2) To build with arm-linux-gnueabihf.toolchain.cmake
SET(TENGINE_TOOLCHAIN_FLAG "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4")
SET(TENGINE_FORCE_SKIP_OPENMP OFF)
3) To build with mips64-linux-gnu.toolchain.cmake
set(CMAKE_C_FLAGS "-march=mips64r2 -mabi=64 -mmsa -mhard-float -mfp64")
set(CMAKE_CXX_FLAGS "-march=mips64r2 -mabi=64 -mmsa -mhard-float -mfp64")
4) To cache flags:
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
```

## Quick Build (Android) ##
```
* Note: set TENGINE_BUILD_TESTS/TENGINE_DEBUG_MEM_STAT OFF
$ ANDROID_NDK=/home/qinhj/desktop/android-ndk-r18b
$ ANDROID_VER=android-28

## Android: Arm64
$ mkdir -p build.android-aarch64 && cd build.android-aarch64
$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=$ANDROID_VER ..
$ make && make install

## Android: Arm32
$ mkdir -p build.android-armv7 && cd build.android-armv7
$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=$ANDROID_VER .. # -DCMAKE_C_FLAGS="-std=gnu99"
$ make && make install
```

## Quick Benchmark ##
```
$ cd build
$ ln -sf ../benchmark/models
$ export LD_LIBRARY_PATH=install/lib:$LD_LIBRARY_PATH
$ ./benchmark/tm_benchmark
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

2. How to check COLLECT_GCC_OPTIONS and IS?
A: "echo 'main(){}' | ${CROSS_TOOLS}gcc -E -v -" and "cat /proc/cpuinfo".

3. .../tengine/source/system/cpu.c:336: undefined reference to `__kmpc_global_thread_num'
A: try another ndk toolchain, or don't "make install" while building with static library.
Otherwise, add necessary "libomp.a" by hand.
```
