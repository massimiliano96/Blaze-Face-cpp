#include "ImagePreparatorImpl.hpp"

ImagePreparatorImpl::ImagePreparatorImpl(int width, int height) : ImagePreparator(), inputWidth(width), inputHeight(height) {};

cv::Mat ImagePreparatorImpl::prepare(const cv::Mat image) const
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

    // Input values should be from -1 to 1 with a size of 128 x 128 pixels
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    cv::resize(img, img, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_CUBIC);
    img = (img - 0.5) / 0.5;

    // Adjust matrix dimensions
    cv::Mat preparedImage;
    img.convertTo(preparedImage, CV_32F);
    preparedImage = preparedImage.reshape(1, inputHeight);
    preparedImage = preparedImage.reshape(1, inputWidth * inputHeight * channels);
    return preparedImage;
}