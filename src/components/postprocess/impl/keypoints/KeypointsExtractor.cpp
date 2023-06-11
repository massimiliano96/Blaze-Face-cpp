#include "KeypointsExtractor.hpp"
#include "utils/anchors_generator/AnchorsGenerator.hpp"

const int KEY_POINT_SIZE = 6;

KeypointsExtractor::KeypointsExtractor(int inputHeight, int inputWidth, const std::vector<Anchor>& anchors)
    : DetectionDataExtractor(), inputHeight(inputHeight), inputWidth(inputWidth), anchors(anchors)
{
}

std::vector<cv::Point2f> KeypointsExtractor::extract(const cv::Mat& rawData) const
{
    std::vector<cv::Point2f> output;
    for (int i = 0; i < indices.size(); i++)
    {
        const int& detectionIdx = indices[i];
        const Anchor& anchor = anchors[detectionIdx];
        for (int j = 0; j < KEY_POINT_SIZE; j++)
        {
            float lx = rawData.at<float>(detectionIdx, 4 + (2 * j) + 0);
            float ly = rawData.at<float>(detectionIdx, 4 + (2 * j) + 1);
            lx += anchor.getX() * (float)inputWidth;
            ly += anchor.getY() * (float)inputHeight;
            lx /= (float)inputWidth;
            ly /= (float)inputHeight;
            output.emplace_back(lx * (float)imageWidth, ly * float(imageHeight));
        }
    }
    return output;
}

void KeypointsExtractor::setParameters(const int width, const int height, const std::vector<int>& inputIndices)
{
    this->imageHeight = height;
    this->imageWidth = width;
    this->indices = inputIndices;
}