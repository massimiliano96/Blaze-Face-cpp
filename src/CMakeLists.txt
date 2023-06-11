add_subdirectory(utils)
add_subdirectory(components)
add_subdirectory(blazeface)
add_subdirectory(factory)

add_library(${LIBRARY_NAME} STATIC
    ${ANCHORS_SOURCES}
    ${ANCHORS_GENERATOR_SOURCES}
    ${PREPROCESS_SOURCES}
    ${FRONT_FACE_SOURCES}
    ${POSTPROCESS_SOURCES}
    ${BLAZE_FACE_SOURCES}
    ${FACTORY_SOURCES}
)

target_sources(${LIBRARY_NAME}
    PUBLIC FILE_SET headers
    TYPE HEADERS
    BASE_DIRS "" FILES
    ${ANCHORS_HEADERS}
    ${ANCHORS_GENERATOR_HEADERS}
    ${PREPROCESS_HEADERS}
    ${FRONT_FACE_HEADERS}
    ${POSTPROCESS_HEADERS}
    ${BLAZE_FACE_HEADERS}
    ${FACTORY_HEADERS}
)
target_link_libraries(${LIBRARY_NAME}
    tensorflow::tensorflowlite
    opencv::opencv
)