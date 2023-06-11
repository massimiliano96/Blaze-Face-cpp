#include "BlazeFaceImpl.hpp"

#include <opencv2/dnn/dnn.hpp>

const int KEY_POINT_SIZE = 6;

BlazeFaceImpl::BlazeFaceImpl(std::shared_ptr<ImagePreparator> preProcessor, std::shared_ptr<FaceDetector> processor,
                             std::shared_ptr<DetectionDataExtractor<int>> indicesExtractor, std::shared_ptr<DetectionDataExtractor<float>> scoresExtractor,
                             std::shared_ptr<DetectionDataExtractor<cv::Rect>> boxesExtractor,
                             std::shared_ptr<DetectionDataExtractor<cv::Point2f>> keypointsExtractor, float scoreThreshold, float iouThreshold)
    : BlazeFace(), preProcessor(preProcessor), processor(processor), indicesExtractor(indicesExtractor), scoresExtractor(scoresExtractor),
      boxesExtractor(boxesExtractor), keypointsExtractor(keypointsExtractor), scoreThreshold(scoreThreshold), iouThreshold(iouThreshold)
{
}

DetectionResults BlazeFaceImpl::detectFaces(const cv::Mat& image) const
{
    DetectionResults output;
    cv::Mat preparedImage = preProcessor->prepare(image);
    RawData detections = processor->inference(preparedImage);

    std::vector<int> indices = indicesExtractor->extract(detections.scores);

    boxesExtractor->setParameters(image.cols, image.rows, indices);
    std::vector<cv::Rect> boxes = boxesExtractor->extract(detections.faces);

    keypointsExtractor->setParameters(image.cols, image.rows, indices);
    std::vector<cv::Point2f> keypoints = keypointsExtractor->extract(detections.faces);

    scoresExtractor->setParameters(image.cols, image.rows, indices);
    std::vector<float> scores = scoresExtractor->extract(detections.scores);

    std::vector<int> filteredIndices;
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, iouThreshold, filteredIndices);

    for (const auto& index : filteredIndices)
    {
        output.boxes.push_back(boxes[index]);
        output.scores.push_back(scores[index]);
        for (int i = 0; i < KEY_POINT_SIZE; i++)
            output.keypoints.push_back(keypoints[index * KEY_POINT_SIZE + i]);
    }

    return output;
}