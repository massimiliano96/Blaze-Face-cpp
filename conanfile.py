import os

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps
from conan.tools.layout import cmake_layout


class CompressorRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    
    def requirements(self):
        self.requires("opencv/4.5.5")
        self.requires("tensorflow-lite/2.10.0")
    
    tools_requires= "pkgconf/1.7.4"

    default_options = {
        "opencv:shared": False,
        "opencv:with_gtk": False,
        "opencv:with_v4l": False,
        "opencv:with_ffmpeg": False,
    }

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()
    
    def layout(self):
        cmake_layout(self)