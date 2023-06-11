#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct DataExtrationInfo
{
    // Model Input dims
    int inputHeight;
    int inputWidth;
};

template <typename T>
class DetectionDataExtractor
{
public:
    DetectionDataExtractor() = default;
    virtual ~DetectionDataExtractor() = default;
    virtual std::vector<T> extract(const cv::Mat& rawData) const = 0;
    virtual void setParameters(const int imageWidth, const int imageHeight, const std::vector<int>& indices) {};
};