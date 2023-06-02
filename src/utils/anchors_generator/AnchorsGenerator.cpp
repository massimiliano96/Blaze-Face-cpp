#include "AnchorsGenerator.hpp"

std::vector<Anchor> AnchorsGenerator::generateAnchors(const SsdAnchorsCalculatorOptions& options)
{
    std::vector<Anchor> anchors;

    // Verify the options.
    if (options.stridesSize != options.numLayers)
    {
        std::cout << "stridesSize and numLayers must be equal." << std::endl;
        return anchors;
    }

    int layerId = 0;
    while (layerId < options.stridesSize)
    {
        std::vector<float> anchorHeight;
        std::vector<float> anchorWidth;
        std::vector<float> aspectRatios;
        std::vector<float> scales;

        // For same strides, we merge the anchors in the same order.
        int lastSameStrideLayer = layerId;
        while (lastSameStrideLayer < options.stridesSize && options.strides[lastSameStrideLayer] == options.strides[layerId])
        {
            float scale = options.minScale + (options.maxScale - options.minScale) * 1.0 * lastSameStrideLayer / (options.stridesSize - 1.0);
            if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer)
            {
                // For the first layer, it can be specified to use predefined anchors.
                aspectRatios = {1.0, 2.0, 0.5};
                scales = {0.1, scale, scale};
            }
            else
            {
                for (size_t aspectRatioId = 0; aspectRatioId < options.aspectRatiosSize; ++aspectRatioId)
                {
                    aspectRatios.push_back(options.aspectRatios[aspectRatioId]);
                    scales.push_back(scale);
                }

                if (options.interpolatedScaleAspectRatio > 0.0)
                {
                    float scaleNext =
                        (lastSameStrideLayer == options.stridesSize - 1)
                            ? 1.0
                            : options.minScale + (options.maxScale - options.minScale) * 1.0 * (lastSameStrideLayer + 1) / (options.stridesSize - 1.0);
                    scales.push_back(std::sqrt(scale * scaleNext));
                    aspectRatios.push_back(options.interpolatedScaleAspectRatio);
                }
            }
            lastSameStrideLayer++;
        }

        for (size_t i = 0; i < aspectRatios.size(); ++i)
        {
            float ratioSqrts = std::sqrt(aspectRatios[i]);
            anchorHeight.push_back(scales[i] / ratioSqrts);
            anchorWidth.push_back(scales[i] * ratioSqrts);
        }

        int featureMapHeight = 0;
        int featureMapWidth = 0;
        if (options.featureMapHeightSize > 0)
        {
            featureMapHeight = options.featureMapHeight[layerId];
            featureMapWidth = options.featureMapWidth[layerId];
        }
        else
        {
            int stride = options.strides[layerId];
            featureMapHeight = std::ceil(1.0 * options.inputSizeHeight / stride);
            featureMapWidth = std::ceil(1.0 * options.inputSizeWidth / stride);
        }

        for (int y = 0; y < featureMapHeight; ++y)
        {
            for (int x = 0; x < featureMapWidth; ++x)
            {
                for (size_t anchorId = 0; anchorId < anchorHeight.size(); ++anchorId)
                {
                    // TODO: Support specifying anchorOffsetX, anchorOffsetY.
                    float xCenter = (x + options.anchorOffsetX) * 1.0 / featureMapWidth;
                    float yCenter = (y + options.anchorOffsetY) * 1.0 / featureMapHeight;
                    float anchorWidthValue = 0.0;
                    float anchorHeightValue = 0.0;
                    if (options.fixedAnchorSize)
                    {
                        anchorWidthValue = 1.0;
                        anchorHeightValue = 1.0;
                    }
                    else
                    {
                        anchorWidthValue = anchorWidth[anchorId];
                        anchorHeightValue = anchorHeight[anchorId];
                    }
                    Anchor newAnchor(xCenter, yCenter, anchorHeightValue, anchorWidthValue);
                    anchors.push_back(newAnchor);
                }
            }
        }
        layerId = lastSameStrideLayer;
    }
    return anchors;
}