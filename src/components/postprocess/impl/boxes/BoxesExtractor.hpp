#pragma once

#include "components/postprocess/DetectionDataExtractor.hpp"
#include "utils/anchors/Anchors.hpp"

class BoxesExtractor : public DetectionDataExtractor<cv::Rect>
{
public:
    explicit BoxesExtractor(int inputHeight, int inputWidth, const std::vector<Anchor>& anchors);
    std::vector<cv::Rect> extract(const cv::Mat& rawData) const override;
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