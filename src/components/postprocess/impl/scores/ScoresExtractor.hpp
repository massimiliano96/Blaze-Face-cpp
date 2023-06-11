#pragma once

#include "components/postprocess/DetectionDataExtractor.hpp"

class ScoresExtractor : public DetectionDataExtractor<float>
{
public:
    explicit ScoresExtractor();
    std::vector<float> extract(const cv::Mat& rawData) const override;
    void setParameters(const int width, const int height, const std::vector<int>& indices) override;

private:
    std::vector<int> indices;
};