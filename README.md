# Blaze-Face-cpp
C++ implementation of the Blaze Face tflite model

## Requirements
- conan (1.60.0) : `pip3 install conan==1.60.0`
- cmake (>=3.23.0) : https://apt.kitware.com/

## Dependencies installation
### Release 
`conan install . --build=missing -s build_type=Release`
### Debug 
`conan install . --build=missing -s build_type=Debug`

## Build
### Targets
- Detection library : `cmake --build build/<build_type> --target blazeFace`
- Demo executable: `cmake --build build/<build_type> --target Executable`