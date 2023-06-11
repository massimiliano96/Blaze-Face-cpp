#pragma once

#include "components/postprocess/DetectionDataExtractor.hpp"
#include "utils/anchors/Anchors.hpp"

class KeypointsExtractor : public DetectionDataExtractor<cv::Point2f>
{
public:
    explicit KeypointsExtractor(int inputHeight, int inputWidth, const std::vector<Anchor>& anchors);
    std::vector<cv::Point2f> extract(const cv::Mat& rawData) const override;
    void setParameters(const int width, const int height, const std::vector<int>& indices) override;

private:
    // Image dims
    int imageHeight;
    int imageWidth;

    // Input dims
    int inputHeight;
    int inputWidth;

    // Anchors
    std::vector<Anchor> anchors;

    // Filtered Indices
    std::vector<int> indices;
};