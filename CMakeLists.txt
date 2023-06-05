cmake_minimum_required(VERSION 3.23)

project(CppProjectTemplate VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/generators/conan_toolchain.cmake)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options("-g")
else()
    add_compile_options("-O3")
endif()

option(ENABLE_SANITIZE_ADDR "Enable address sanitize." ON)
option(ENABLE_SANITIZE_UNDEF "Enable undefined sanitize." ON)
option(ENABLE_SANITIZE_LEAK "Enable leak sanitize (Gcc/Clang only)." ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

if(ENABLE_SANITIZE_ADDR OR ENABLE_SANITIZE_UNDEF)
    include(Sanitizers)
    add_sanitizer_flags()
endif()

find_package(OpenCV)
find_package(tensorflowlite)

set(LIBRARY_NAME blazeface)

file(COPY models DESTINATION ./app/)
file(COPY data DESTINATION ./app/)

add_subdirectory(src)
add_subdirectory(app)