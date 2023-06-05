#pragma once

#include "components/detection/FaceDetector.hpp"

#include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

class FrontFaceDetector : public FaceDetector
{
public:
    FrontFaceDetector(std::string& modelFile, float scoreThreshold, float iouThreshold);
    RawData inference(const cv::Mat& inputTensor) override;
    ModelInputDetails getModelInputDetails() const override;

private:
    std::unique_ptr<tflite::Interpreter> interpreter;

    float scoreThreshold;
    float iouThreshold;
    float sigmoidScoreThreshold;
}