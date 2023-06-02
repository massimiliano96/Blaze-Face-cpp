#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "faceDetecor/FaceDetector.hpp"

int main()
{
    BlazeFaceDetector detector;

    cv::Mat image = cv::imread("data/faces.jpg");

    detector.initializeModel();

    DetectionResults output = detector.detectFaces(image);

    for (size_t i = 0; i < output.boxes.size(); i++)
    {
        cv::rectangle(image, output.boxes[i], cv::Scalar(0, 255, 0));
    }

    for (size_t i = 0; i < output.keypoints.size(); i++)
    {
        cv::circle(image, output.keypoints[i], 3, cv::Scalar(255, 0, 0));
    }

    cv::imwrite("./data/detection.jpg", image);

    return 0;
}
