#pragma once

#include "components/postprocess/DetectionDataExtractor.hpp"

class IndicesExtractor : public DetectionDataExtractor<int>
{
public:
    explicit IndicesExtractor(float sigmoidScoreThreshold);
    std::vector<int> extract(const cv::Mat& rawData) const override;

private:
    float sigmoidScoreThreshold;
};