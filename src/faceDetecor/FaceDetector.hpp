#include <cmath>
#include <iostream>
#include <vector>

#include "utils/anchors/Anchors.hpp"
#include "utils/anchors_generator/AnchorsGenerator.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <opencv2/core.hpp>

#include "Parameters.hpp"

const int KEY_POINT_SIZE = 6;
const int MAX_FACE_NUM = 100;

struct DetectionResults
{
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> scores;
};

class BlazeFaceDetector
{
public:
    BlazeFaceDetector(float scoreThreshold = 0.7, float iouThreshold = 0.3);

    void initializeModel();

    DetectionResults detectFaces(cv::Mat& image);

private:
    void getModelInputDetails();

    void generateAnchors();

    cv::Mat prepareInputForInference(const cv::Mat& image);

    void inference(const cv::Mat& inputTensor, cv::Mat& faces, cv::Mat& scores);

    DetectionResults extractDetections(const cv::Mat& rawBoxes, cv::Mat& rawScores, const std::vector<int>& goodDetectionsIndices);

    std::vector<int> filterDetections(const cv::Mat& output1);

    DetectionResults filterWithNonMaxSupression(std::vector<cv::Rect>& boxes, std::vector<cv::Point2f>& keypoints, std::vector<float>& scores);

    std::unique_ptr<tflite::Interpreter> interpreter;
    std::string type;
    float scoreThreshold;
    float iouThreshold;
    float sigmoidScoreThreshold;

    // Model input details
    int inputHeight;
    int inputWidth;
    int channels;

    // Image dims
    int imageHeight;
    int imageWidth;

    // Anchors
    std::vector<Anchor> anchors;
};