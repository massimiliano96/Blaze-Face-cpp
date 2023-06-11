#include "FrontBlazeFaceFactory.hpp"
#include "Parameters.hpp"
#include "blazeface/impl/BlazeFaceImpl.hpp"
#include "components/detection/FaceDetector.hpp"
#include "components/detection/impl/frontFace/FrontFaceDetector.hpp"
#include "components/postprocess/DetectionDataExtractor.hpp"
#include "components/postprocess/impl/boxes/BoxesExtractor.hpp"
#include "components/postprocess/impl/indices/IndicesExtractor.hpp"
#include "components/postprocess/impl/keypoints/KeypointsExtractor.hpp"
#include "components/postprocess/impl/scores/ScoresExtractor.hpp"
#include "components/preprocess/ImagePreparator.hpp"
#include "components/preprocess/impl/ImagePreparatorImpl.hpp"
#include "utils/anchors_generator/AnchorsGenerator.hpp"

FrontBlazeFaceFactory::FrontBlazeFaceFactory() : BlazeFaceFactory() {};

std::shared_ptr<BlazeFace> FrontBlazeFaceFactory::create(std::string& modelFile, float scoreThreshold, float iouThreshold)
{
    SsdAnchorsCalculatorOptions anchorsOption;
    anchorsOption = {inputSizeWidth,
                     inputSizeHeight,
                     minScale,
                     maxScale,
                     anchorOffsetX,
                     anchorOffsetY,
                     numLayers,
                     featureMapWidth,
                     featureMapHeight,
                     featureMapWidthSize,
                     featureMapHeightSize,
                     strides,
                     stridesSize,
                     aspectRatios,
                     aspectRatiosSize,
                     reduceBoxesInLowestLayer,
                     interpolatedScaleAspectRatio,
                     fixedAnchorSize};
    std::vector<Anchor> anchors = AnchorsGenerator::generateAnchors(anchorsOption);

    std::shared_ptr<FaceDetector> processor = std::make_shared<FrontFaceDetector>(modelFile);
    ModelInputDetails inputDetails = processor->getModelInputDetails();

    std::shared_ptr<ImagePreparator> preprocessor =
        std::make_shared<ImagePreparatorImpl>(inputDetails.inputWidth, inputDetails.inputHeight, inputDetails.inputChannels);

    float sigmoidThreshold = std::log(scoreThreshold / (1 - scoreThreshold));
    std::shared_ptr<DetectionDataExtractor<int>> indicesExtractor = std::make_shared<IndicesExtractor>(sigmoidThreshold);
    std::shared_ptr<DetectionDataExtractor<cv::Rect>> boxesExtractor =
        std::make_shared<BoxesExtractor>(inputDetails.inputHeight, inputDetails.inputWidth, anchors);
    std::shared_ptr<DetectionDataExtractor<cv::Point2f>> keypointsExtractor =
        std::make_shared<KeypointsExtractor>(inputDetails.inputHeight, inputDetails.inputWidth, anchors);
    std::shared_ptr<DetectionDataExtractor<float>> scoresExtractor = std::make_shared<ScoresExtractor>();

    std::shared_ptr<BlazeFace> blazeFace = std::make_shared<BlazeFaceImpl>(preprocessor, processor, indicesExtractor, scoresExtractor, boxesExtractor,
                                                                           keypointsExtractor, scoreThreshold, iouThreshold);

    return blazeFace;
}
