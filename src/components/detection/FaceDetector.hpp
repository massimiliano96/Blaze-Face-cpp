#pragma once

#include <opencv2/core.hpp>

struct RawData
{
    cv::Mat faces;
    cv::Mat scores;
};

struct ModelInputDetails
{
    int inputHeight;
    int inputWidth;
    int inputChannels;
}

class FaceDetector
{
public:
    FaceDetector();
    virtual ~FaceDetector();
    virtual RawData inference(const cv::Mat& inputTensor) = 0;
    virtual ModelInputDetails getModelInputDetails() const = 0;
};