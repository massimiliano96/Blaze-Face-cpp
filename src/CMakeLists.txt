add_subdirectory(utils)
add_subdirectory(faceDetecor)

add_library(${LIBRARY_NAME} STATIC
    ${ANCHORS_SOURCES}
    ${ANCHORS_GENERATOR_SOURCES}
    ${FACE_DETECTOR_SOURCES}
)

target_sources(${LIBRARY_NAME}
    PUBLIC FILE_SET headers
    TYPE HEADERS
    BASE_DIRS "" FILES
    ${ANCHORS_HEADERS}
    ${ANCHORS_GENERATOR_HEADERS}
    ${FACE_DETECTOR_HEADERS}
)
target_link_libraries(${LIBRARY_NAME}
    tensorflow::tensorflowlite
    opencv::opencv
)