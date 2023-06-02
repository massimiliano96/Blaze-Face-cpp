#pragma once

#include <cstdio>
#include <vector>

const int inputSizeWidth = 128;
const int inputSizeHeight = 128;
const float minScale = 0.1484375;
const float maxScale = 0.75;
const float anchorOffsetX = 0.5;
const float anchorOffsetY = 0.5;
const int numLayers = 4;
const std::vector<int> featureMapWidth = {};
const std::vector<int> featureMapHeight = {};
const size_t featureMapWidthSize = 0;
const size_t featureMapHeightSize = 0;
const std::vector<int> strides = {8, 16, 16, 16};
const size_t stridesSize = 4;
const std::vector<float> aspectRatios = {1.0};
const size_t aspectRatiosSize = 1;
const bool reduceBoxesInLowestLayer = false;
const float interpolatedScaleAspectRatio = 1.0;
const bool fixedAnchorSize = true;