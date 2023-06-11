#include "BoxesExtractor.hpp"

BoxesExtractor::BoxesExtractor(int inputHeight, int inputWidth, const std::vector<Anchor>& anchors)
    : DetectionDataExtractor(), inputHeight(inputHeight), inputWidth(inputWidth), anchors(anchors)
{
}

std::vector<cv::Rect> BoxesExtractor::extract(const cv::Mat& rawData) const
{
    std::vector<cv::Rect> output;
    for (int i = 0; i < indices.size(); i++)
    {
        const int& detectionIdx = indices[i];
        const Anchor& anchor = anchors[detectionIdx];

        float sx = rawData.at<float>(detectionIdx, 0);
        float sy = rawData.at<float>(detectionIdx, 1);
        float w = rawData.at<float>(detectionIdx, 2);
        float h = rawData.at<float>(detectionIdx, 3);

        float cx = sx + anchor.getX() * (float)inputWidth;
        float cy = sy + anchor.getY() * (float)inputHeight;

        cx /= (float)inputWidth;
        cy /= (float)inputHeight;
        w /= (float)inputWidth;
        h /= (float)inputHeight;

        auto x = int((cx - w * 0.5) * imageWidth);
        auto y = int((cy - h * 0.5) * imageHeight);
        auto width = int(w * float(imageWidth));
        auto height = int(h * float(imageHeight));
        output.emplace_back(x, y, width, height);
    }
    return output;
}

void BoxesExtractor::setParameters(const int width, const int height, const std::vector<int>& inputIndices)
{
    this->imageHeight = height;
    this->imageWidth = width;
    this->indices = inputIndices;
}