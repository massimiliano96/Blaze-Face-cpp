cmake_minimum_required(VERSION 3.23)

project(CppProjectTemplate VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/generators/conan_toolchain.cmake)

add_compile_options("-g")

find_package(OpenCV)
find_package(tensorflowlite)

set(LIBRARY_NAME blazeface)

file(COPY models DESTINATION ./app/)
file(COPY data DESTINATION ./app/)

add_subdirectory(src)
add_subdirectory(app)