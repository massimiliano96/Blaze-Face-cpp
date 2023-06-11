#pragma once

#include "blazeface/BlazeFace.hpp"

class BlazeFaceFactory
{
public:
    BlazeFaceFactory() = default;
    virtual ~BlazeFaceFactory() = default;
    virtual std::shared_ptr<BlazeFace> create(std::string& modelFile, float scoreThreshold, float iouThreshold) = 0;
};