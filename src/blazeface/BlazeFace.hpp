#pragma once

#include <opencv2/core.hpp>

struct DetectionResults
{
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> scores;
};

class BlazeFace
{
public:
    BlazeFace() = default;
    virtual ~BlazeFace() = default;
    virtual DetectionResults detectFaces(const cv::Mat& image) const = 0;
};