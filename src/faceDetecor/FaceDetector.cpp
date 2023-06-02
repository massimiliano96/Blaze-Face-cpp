#include "FaceDetector.hpp"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#define MODEL_FILE "models/face_detection_front.tflite"

BlazeFaceDetector::BlazeFaceDetector(float scoreThreshold, float iouThreshold) : scoreThreshold(scoreThreshold), iouThreshold(iouThreshold)
{
    sigmoidScoreThreshold = std::log(scoreThreshold / (1 - scoreThreshold));

    // Initialize model based on model type
    initializeModel();

    // Generate anchors for model
    generateAnchors();
}

void BlazeFaceDetector::generateAnchors()
{
    SsdAnchorsCalculatorOptions anchorsOption;
    anchorsOption = {inputSizeWidth,
                     inputSizeHeight,
                     minScale,
                     maxScale,
                     anchorOffsetX,
                     anchorOffsetY,
                     numLayers,
                     featureMapWidth,
                     featureMapHeight,
                     featureMapWidthSize,
                     featureMapHeightSize,
                     strides,
                     stridesSize,
                     aspectRatios,
                     aspectRatiosSize,
                     reduceBoxesInLowestLayer,
                     interpolatedScaleAspectRatio,
                     fixedAnchorSize};
    anchors = AnchorsGenerator::generateAnchors(anchorsOption);
}

void BlazeFaceDetector::initializeModel()
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILE);
    // Initiate Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    assert(interpreter != nullptr);
    interpreter->AllocateTensors();
    assert(interpreter->AllocateTensors() == kTfLiteOk);

    // Get model info
    getModelInputDetails();
    // getModelOutputDetails();
}

DetectionResults BlazeFaceDetector::detectFaces(cv::Mat& image)
{
    imageHeight = image.rows;
    imageWidth = image.cols;

    // Prepare image for inference
    auto inputTensor = prepareInputForInference(image);

    cv::Mat rawBoxes;
    cv::Mat rawScores;
    // Perform inference on the image
    inference(inputTensor, rawBoxes, rawScores);

    // Filter scores based on the detection scores
    std::vector<int> goodDetectionIndeces = {};
    goodDetectionIndeces = filterDetections(rawScores);

    // Extract information of filtered detections
    DetectionResults extractedOutput = extractDetections(rawBoxes, rawScores, goodDetectionIndeces);
    std::vector<cv::Point2f> keypoints = extractedOutput.keypoints;
    std::vector<cv::Rect> boxes = extractedOutput.boxes;
    std::vector<float> scores = extractedOutput.scores;

    // Filter results with non-maximum suppression
    DetectionResults filteredOutput = filterWithNonMaxSupression(boxes, keypoints, scores);

    return filteredOutput;
}

void BlazeFaceDetector::getModelInputDetails()
{
    int inputIdx = interpreter->inputs()[0];
    TfLiteTensor* inputTensor = interpreter->tensor(inputIdx);
    TfLiteIntArray* inputDims = inputTensor->dims;
    inputHeight = inputDims->data[1];
    inputWidth = inputDims->data[2];
    channels = inputDims->data[3];
}

// void BlazeFaceDetector::getModelOutputDetails()
// {
//     const auto& output_details = interpreter->get_input_details();
//     output0_index = output_details[0].index;
//     output1_index = output_details[1].index;
// }

cv::Mat BlazeFaceDetector::prepareInputForInference(const cv::Mat& image)
{
    cv::Mat img;
    // OpenCV images are in BGR, model expects RGB channel format.
    int cnls = image.type();
    if (cnls == CV_8UC4)
    {
        cvtColor(image, img, cv::COLOR_BGRA2RGB);
        std::cout << "This is CV_8UC4 image format" << std::endl;
    }
    else if (cnls == CV_8UC3)
    {
        cvtColor(image, img, cv::COLOR_BGR2RGB);
        std::cout << "This is CV_8UC3 image format" << std::endl;
    }
    else
    {
        std::cout << "Image format is not supported" << std::endl;
        throw std::runtime_error("Image format is not supported");
    }

    // Input values should be from -1 to 1 with a size of 128 x 128 pixels for
    // the front model and 256 x 256 pixels for the back model
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    cv::resize(img, img, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_CUBIC);
    img = (img - 0.5) / 0.5;

    // Adjust matrix dimensions
    cv::Mat tensor;
    img.convertTo(tensor, CV_32F);
    tensor = tensor.reshape(1, inputHeight);
    tensor = tensor.reshape(1, inputWidth * inputHeight * channels);
    return tensor;
}

void BlazeFaceDetector::inference(const cv::Mat& inputTensor, cv::Mat& faces, cv::Mat& scores)
{
    // Input Tensor
    int input = interpreter->inputs()[0];
    TfLiteTensor* tensor = interpreter->tensor(input);
    float* dst = tensor->data.f;
    memcpy(dst, inputTensor.data, inputTensor.total() * inputTensor.elemSize());

    // Prepare output tensors
    // Faces
    int facesOutput = interpreter->outputs()[0];
    TfLiteTensor* facesOutputTensor = interpreter->tensor(facesOutput);
    TfLiteIntArray* facesOutputDims = interpreter->tensor(facesOutput)->dims;
    int facesRows = facesOutputDims->data[facesOutputDims->size - 2];
    int facesCols = facesOutputDims->data[facesOutputDims->size - 1];
    float* facesData = facesOutputTensor->data.f;
    faces.create(facesRows, facesCols, CV_32F);

    // Scores
    int scoresOutput = interpreter->outputs()[1];
    TfLiteTensor* scoresOutput_tensor = interpreter->tensor(scoresOutput);
    TfLiteIntArray* scoresOutputDims = interpreter->tensor(scoresOutput)->dims;
    int scoresRows = scoresOutputDims->data[scoresOutputDims->size - 2];
    int scoresCols = scoresOutputDims->data[scoresOutputDims->size - 1];
    float* scoresData = scoresOutput_tensor->data.f;
    scores.create(scoresRows, scoresCols, CV_32F);

    // Inference
    interpreter->Invoke();

    memcpy(faces.data, facesData, sizeof(float) * facesRows * facesCols);
    memcpy(scores.data, scoresData, sizeof(float) * scoresRows * scoresCols);
}

DetectionResults BlazeFaceDetector::extractDetections(const cv::Mat& rawBoxes, cv::Mat& rawScores, const std::vector<int>& goodDetectionsIndices)
{
    DetectionResults output;
    std::size_t numGoodDetections = goodDetectionsIndices.size();

    for (int i = 0; i < numGoodDetections; i++)
    {
        const int& detectionIdx = goodDetectionsIndices[i];
        float currentScore = 1.0 / (1.0 + std::exp(-rawScores.at<float>(detectionIdx)));
        output.scores.push_back(currentScore);
        const Anchor& anchor = anchors[detectionIdx];

        float sx = rawBoxes.at<float>(detectionIdx, 0);
        float sy = rawBoxes.at<float>(detectionIdx, 1);
        float w = rawBoxes.at<float>(detectionIdx, 2);
        float h = rawBoxes.at<float>(detectionIdx, 3);

        float cx = sx + anchor.getX() * (float)inputWidth;
        float cy = sy + anchor.getY() * (float)inputHeight;

        cx /= (float)inputWidth;
        cy /= (float)inputHeight;
        w /= (float)inputWidth;
        h /= (float)inputHeight;

        for (int j = 0; j < KEY_POINT_SIZE; j++)
        {
            float lx = rawBoxes.at<float>(detectionIdx, 4 + (2 * j) + 0);
            float ly = rawBoxes.at<float>(detectionIdx, 4 + (2 * j) + 1);
            lx += anchor.getX() * (float)inputWidth;
            ly += anchor.getY() * (float)inputHeight;
            lx /= (float)inputWidth;
            ly /= (float)inputHeight;
            output.keypoints.emplace_back(lx * imageWidth, ly * imageHeight);
        }
        int x = (cx - w * 0.5) * imageWidth;
        int y = (cy - h * 0.5) * imageHeight;
        int width = w * imageWidth;
        int height = h * imageHeight;
        output.boxes.emplace_back(x, y, width, height);
    }
    return output;
}

std::vector<int> BlazeFaceDetector::filterDetections(const cv::Mat& output1)
{
    // Filter based on the score threshold before applying the sigmoid function
    std::vector<int> goodDetections;
    for (int i = 0; i < output1.rows; i++)
    {
        float score = output1.at<float>(i);
        if (score > sigmoidScoreThreshold)
        {
            goodDetections.push_back(i);
        }
    }

    return goodDetections;
}

DetectionResults BlazeFaceDetector::filterWithNonMaxSupression(std::vector<cv::Rect>& boxes, std::vector<cv::Point2f>& keypoints, std::vector<float>& scores)
{
    DetectionResults output;

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, iouThreshold, indices);

    for (const auto& index : indices)
    {
        output.boxes.push_back(boxes[index]);
        output.scores.push_back(scores[index]);
        for (int i = 0; i < KEY_POINT_SIZE; i++)
            output.keypoints.push_back(keypoints[index + i]);
    }

    return output;
}
