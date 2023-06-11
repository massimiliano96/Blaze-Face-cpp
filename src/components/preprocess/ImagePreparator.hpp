#pragma once

#include <opencv2/core.hpp>

class ImagePreparator
{
public:
    ImagePreparator() = default;
    virtual ~ImagePreparator() = default;
    virtual cv::Mat prepare(const cv::Mat& image) const = 0;
};