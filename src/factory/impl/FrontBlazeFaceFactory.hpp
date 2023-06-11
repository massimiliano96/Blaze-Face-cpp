#pragma once

#include "factory/BlazeFaceFactory.hpp"

class FrontBlazeFaceFactory : public BlazeFaceFactory
{
public:
    FrontBlazeFaceFactory();
    std::shared_ptr<BlazeFace> create(std::string& modelFile, float scoreThreshold, float iouThreshold) override;
};