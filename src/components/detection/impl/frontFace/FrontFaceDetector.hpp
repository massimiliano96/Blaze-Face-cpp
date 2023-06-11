#pragma once

#include "components/detection/FaceDetector.hpp"

#include "tensorflow/lite/interpreter.h"

class FrontFaceDetector : public FaceDetector
{
public:
    explicit FrontFaceDetector(std::string& modelFile);
    RawData inference(const cv::Mat& inputTensor) override;
    ModelInputDetails getModelInputDetails() const override;

private:
    std::unique_ptr<tflite::Interpreter> interpreter;
};