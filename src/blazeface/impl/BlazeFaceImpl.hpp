#pragma once

#include "blazeface/BlazeFace.hpp"

#include "components/detection/FaceDetector.hpp"
#include "components/postprocess/DetectionDataExtractor.hpp"
#include "components/preprocess/ImagePreparator.hpp"

#include <memory>

class BlazeFaceImpl : public BlazeFace
{
public:
    BlazeFaceImpl(std::shared_ptr<ImagePreparator> preProcessor, std::shared_ptr<FaceDetector> processor,
                  std::shared_ptr<DetectionDataExtractor<int>> indicesExtractor, std::shared_ptr<DetectionDataExtractor<float>> scoresExtractor,
                  std::shared_ptr<DetectionDataExtractor<cv::Rect>> boxesExtractor, std::shared_ptr<DetectionDataExtractor<cv::Point2f>> keypointsExtractor,
                  float scoreThreshold, float iouThreshold);

    DetectionResults detectFaces(const cv::Mat& image) const override;

private:
    std::shared_ptr<ImagePreparator> preProcessor;
    std::shared_ptr<FaceDetector> processor;
    std::shared_ptr<DetectionDataExtractor<int>> indicesExtractor;
    std::shared_ptr<DetectionDataExtractor<float>> scoresExtractor;
    std::shared_ptr<DetectionDataExtractor<cv::Rect>> boxesExtractor;
    std::shared_ptr<DetectionDataExtractor<cv::Point2f>> keypointsExtractor;

    float scoreThreshold;
    float iouThreshold;
};