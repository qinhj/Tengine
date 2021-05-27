# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, IMILAB
# Author: qinhongjie@imilab.com
#

# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)

# macro for adding test
FUNCTION (TENGINE_EXAMPLE_IMILAB name file)
    ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${PROJECT_SOURCE_DIR}/examples/common/tengine_operations.c")

    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")

    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME})

    IF(MSVC)
        TARGET_COMPILE_DEFINITIONS (${name} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:NOMINMAX>)
    ELSE()
        TARGET_LINK_LIBRARIES (${name} PRIVATE m)
    ENDIF()

    INSTALL (TARGETS ${name} DESTINATION bin)
ENDFUNCTION()

## hello world
TENGINE_EXAMPLE_IMILAB (tm_hello_world      tm_hello_world.cpp)
## imilab examples
TENGINE_EXAMPLE_IMILAB (tm_test_imread      tm_test_imread.cpp)
TENGINE_EXAMPLE_IMILAB (tm_mobilenet_ssd_   tm_mobilenet_ssd_.c)
TENGINE_EXAMPLE_IMILAB (tm_retinaface_      tm_retinaface_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov3_          tm_yolov3_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov3_p4p5_     tm_yolov3_p4p5_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov3_tiny_     tm_yolov3_tiny_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov5s_         tm_yolov5s_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov5s_tiny_    tm_yolov5s_tiny_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov5s_p3p4_    tm_yolov5s_p3p4_.cpp)
TENGINE_EXAMPLE_IMILAB (tm_yolov5s_uint8_   tm_yolov5s_uint8_.cpp)

# add examples with opencv
# set(OpenCV_DIR /mnt/d/ubuntu/opencv_install/linux-armv7/lib/cmake/opencv4)
FIND_PACKAGE(OpenCV QUIET)

IF (OpenCV_FOUND AND ${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    # macro for adding examples
    FUNCTION (TENGINE_EXAMPLE_IMILAB_CV name file)
        ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${PROJECT_SOURCE_DIR}/examples/common/tengine_operations.c")

        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")

        TARGET_LINK_LIBRARIES (${name} ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
        INSTALL (TARGETS ${name} DESTINATION bin)
    ENDFUNCTION()

    ## imilab opencv examples
    TENGINE_EXAMPLE_IMILAB_CV (tm_yolov3_cv_        tm_yolov3_cv_.cpp)
    TENGINE_EXAMPLE_IMILAB_CV (tm_yolov5s_cv_       tm_yolov5s_cv_.cpp)
ELSE()
    MESSAGE (WARNING "OpenCV not found, some examples won't be built")
ENDIF()
