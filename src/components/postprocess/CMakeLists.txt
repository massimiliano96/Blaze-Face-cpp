set(POSTPROCESS_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/DetectionDataExtractor.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/indices/IndicesExtractor.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/boxes/BoxesExtractor.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/keypoints/KeypointsExtractor.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/scores/ScoresExtractor.hpp" CACHE INTERNAL "")

set(POSTPROCESS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/impl/indices/IndicesExtractor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/boxes/BoxesExtractor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/keypoints/KeypointsExtractor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/impl/scores/ScoresExtractor.cpp" CACHE INTERNAL "")