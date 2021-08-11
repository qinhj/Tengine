// ============================================================
//                  Imilab Utils: Visual APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/12
// ============================================================

#ifndef __IMI_UTILS_VISUAL_HPP__
#define __IMI_UTILS_VISUAL_HPP__

//#define USE_OPENCV

/* std c++ includes */
#include <vector>
/* imilab includes */
#include "imi_utils_elog.h"     // for: log_xxxx
#include "imi_utils_common.hpp" // for: Object

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

template<typename _Tp>
int imi_utils_objects_draw(const std::vector<_Tp>& objects,
                           cv::Mat& img, int cls, const char* const* labels)
{
    size_t size = objects.size();
    log_echo("detected objects num: %zu\n", size);
    for (size_t i = 0; i < size; i++)
    {
        const _Tp& obj = objects[i];
        if (labels)
        {
            log_echo("[%2d]: %3.3f%%, [(%7.3f, %7.3f), (%7.3f, %7.3f)], %s\n",
                     obj.label, obj.prob * 100, obj.rect.x, obj.rect.y,
                     obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, labels[obj.label]);
        }
        if (-1 != cls && obj.label != cls) continue;
        // draw object box
        cv::rectangle(img, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", labels[obj.label], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (img.cols < x + label_size.width)
            x = img.cols - label_size.width;
        // draw label box
        cv::rectangle(img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        // draw label text
        cv::putText(img, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }
    return 0;
}

#else // !USE_OPENCV

#include "tengine_operations.h" // for: image, draw_box

// @param:  cls[in] target class(-1: all)
template<typename _Tp>
int imi_utils_objects_draw(const std::vector<_Tp>& objects,
                           image& img, int cls, const char* const* labels)
{
    size_t size = objects.size();
    log_echo("detected objects num: %zu\n", size);

    for (size_t i = 0; i < size; i++)
    {
        const _Tp& obj = objects[i];
        if (labels)
        {
            log_echo("[%2d]: %3.3f%%, [(%7.3f, %7.3f), (%7.3f, %7.3f)], %s\n",
                     obj.label, obj.prob * 100, obj.rect.x, obj.rect.y,
                     obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, labels[obj.label]);
        }
        if (-1 == cls || obj.label == cls)
        {
            draw_box(img, obj.rect.x, obj.rect.y,
                     obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, 2, 0, 255, 0);
        }
    }
    return 0;
}

template<typename _Tp>
int imi_utils_faces_draw(const std::vector<_Tp>& faces, image& img)
{
    size_t size = faces.size();
    log_echo("detected faces num: %zu\n", size);

    for (size_t i = 0; i < size; i++)
    {
        const _Tp& face = faces[i];
        log_echo("Face[%2zu]: %3.3f%%, [(%7.3f, %7.3f), (%7.3f, %7.3f)]\n",
                 i, face.prob * 100, face.rect.x, face.rect.y, face.rect.width, face.rect.height);

        draw_box(img, face.rect.x, face.rect.y,
                 face.rect.x + face.rect.width, face.rect.y + face.rect.height, 2, 0, 255, 0);
        for (int l = 0; l < 5; l++)
        {
            draw_circle(img, face.landmark[l].x, face.landmark[l].y, 1, 0, 128, 128);
        }
    }
    return 0;
}

#endif // USE_OPENCV

#endif // !__IMI_UTILS_VISUAL_HPP__
