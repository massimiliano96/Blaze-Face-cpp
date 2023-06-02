#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

struct SsdAnchorsCalculatorOptions
{
    int inputSizeWidth;
    int inputSizeHeight;
    float minScale;
    float maxScale;
    float anchorOffsetX;
    float anchorOffsetY;
    int numLayers;
    std::vector<int> featureMapWidth;
    std::vector<int> featureMapHeight;
    size_t featureMapWidthSize;
    size_t featureMapHeightSize;
    std::vector<int> strides;
    size_t stridesSize;
    std::vector<float> aspectRatios;
    size_t aspectRatiosSize;
    bool reduceBoxesInLowestLayer;
    float interpolatedScaleAspectRatio;
    bool fixedAnchorSize;
};

class Anchor
{
public:
    Anchor(float xCenter, float yCenter, float h, float w);

    std::string serialize() const;

    float getX() const;
    float getY() const;
    float getHeight() const;
    float getWidth() const;

private:
    float xCenter;
    float yCenter;
    float h;
    float w;
};