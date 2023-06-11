#pragma once

#include "components/preprocess/ImagePreparator.hpp"

class ImagePreparatorImpl : public ImagePreparator
{
public:
    ImagePreparatorImpl(int width, int height, int channels);
    cv::Mat prepare(const cv::Mat& image) const override;

private:
    int inputWidth;
    int inputHeight;
    int inputChannels;
};
