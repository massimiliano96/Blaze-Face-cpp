#include "FrontFaceDetector.hpp"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

FrontFaceDetector::FrontFaceDetector(std::string& modelFile) : FaceDetector()
{

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFile.c_str());
    // Initiate Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    assert(interpreter != nullptr);
    interpreter->AllocateTensors();
    assert(interpreter->AllocateTensors() == kTfLiteOk);
}

RawData FrontFaceDetector::inference(const cv::Mat& inputTensor)
{
    RawData rawOutput;
    // Input Tensor
    int input = interpreter->inputs()[0];
    TfLiteTensor* tensor = interpreter->tensor(input);
    float* dst = tensor->data.f;
    memcpy(dst, inputTensor.data, inputTensor.total() * inputTensor.elemSize());

    // Prepare output tensors
    // Boxes and Keypoints
    int facesOutput = interpreter->outputs()[0];
    TfLiteTensor* facesOutputTensor = interpreter->tensor(facesOutput);
    TfLiteIntArray* facesOutputDims = interpreter->tensor(facesOutput)->dims;
    int facesRows = facesOutputDims->data[facesOutputDims->size - 2];
    int facesCols = facesOutputDims->data[facesOutputDims->size - 1];
    float* facesData = facesOutputTensor->data.f;
    rawOutput.faces.create(facesRows, facesCols, CV_32F);

    // Scores
    int scoresOutput = interpreter->outputs()[1];
    TfLiteTensor* scoresOutput_tensor = interpreter->tensor(scoresOutput);
    TfLiteIntArray* scoresOutputDims = interpreter->tensor(scoresOutput)->dims;
    int scoresRows = scoresOutputDims->data[scoresOutputDims->size - 2];
    int scoresCols = scoresOutputDims->data[scoresOutputDims->size - 1];
    float* scoresData = scoresOutput_tensor->data.f;
    rawOutput.scores.create(scoresRows, scoresCols, CV_32F);

    // Inference
    interpreter->Invoke();

    memcpy(rawOutput.faces.data, facesData, sizeof(float) * facesRows * facesCols);
    memcpy(rawOutput.scores.data, scoresData, sizeof(float) * scoresRows * scoresCols);

    return rawOutput;
}

ModelInputDetails FrontFaceDetector::getModelInputDetails() const
{
    int inputIdx = interpreter->inputs()[0];
    TfLiteTensor* inputTensor = interpreter->tensor(inputIdx);
    TfLiteIntArray* inputDims = inputTensor->dims;
    ModelInputDetails inputDetails;
    inputDetails.inputHeight = inputDims->data[1];
    inputDetails.inputWidth = inputDims->data[2];
    inputDetails.inputChannels = inputDims->data[3];
    return inputDetails;
}