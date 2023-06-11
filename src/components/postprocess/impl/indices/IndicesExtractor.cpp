#include "IndicesExtractor.hpp"

IndicesExtractor::IndicesExtractor(float sigmoidScoreThreshold) : DetectionDataExtractor(), sigmoidScoreThreshold(sigmoidScoreThreshold)
{
}

std::vector<int> IndicesExtractor::extract(const cv::Mat& rawData) const
{
    std::vector<int> extractedIndices;
    for (int i = 0; i < rawData.rows; i++)
    {
        float score = rawData.at<float>(i);
        if (score > sigmoidScoreThreshold)
        {
            extractedIndices.push_back(i);
        }
    }

    return extractedIndices;
}