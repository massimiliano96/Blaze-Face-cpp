#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "blazeface/BlazeFace.hpp"
#include "factory/impl/FrontBlazeFaceFactory.hpp"

int main()
{
    std::shared_ptr<BlazeFaceFactory> factory = std::make_shared<FrontBlazeFaceFactory>();
    std::string modelFile = "models/face_detection_front.tflite";
    std::shared_ptr<BlazeFace> detector = factory->create(modelFile, 0.7, 0.3);

    cv::Mat image = cv::imread("data/faces.jpg");

    auto start = std::chrono::high_resolution_clock::now();

    DetectionResults output = detector->detectFaces(image);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " milliseconds." << std::endl;

    for (size_t i = 0; i < output.boxes.size(); i++)
    {
        cv::rectangle(image, output.boxes[i], cv::Scalar(0, 255, 0));
    }

    for (size_t i = 0; i < output.keypoints.size(); i++)
    {
        cv::circle(image, output.keypoints[i], 3, cv::Scalar(255, 0, 0));
        cv::imwrite("./data/detection.jpg", image);
    }

    cv::imwrite("./data/detection.jpg", image);

    return 0;
}
