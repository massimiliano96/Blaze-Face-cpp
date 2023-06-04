#pragma once

#include <opencv2/core.hpp>

class ImagePreparator
{
public:
    ImagePreparator();
    virtual ~ImagePreparator();
    virtual cv::Mat prepare(const cv::Mat& image) const = 0;
};