#include "ScoresExtractor.hpp"

ScoresExtractor::ScoresExtractor() : DetectionDataExtractor()
{
}

std::vector<float> ScoresExtractor::extract(const cv::Mat& rawData) const
{
    std::vector<float> extractedScores;
    for (int i = 0; i < indices.size(); i++)
    {
        const int& detectionIdx = indices[i];
        float currentScore = 1.0 / (1.0 + std::exp(-rawData.at<float>(detectionIdx)));
        extractedScores.push_back(currentScore);
    }
    return extractedScores;
}

void ScoresExtractor::setParameters(const int width, const int height, const std::vector<int>& inputIndices)
{
    this->indices = inputIndices;
}