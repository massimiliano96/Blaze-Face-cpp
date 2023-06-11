find_package(Git)

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GIT_TAG_VERSION)
        string(REGEX REPLACE "^v(.*)" "\\1" CPACK_PACKAGE_VERSION "${GIT_TAG_VERSION}")
        message(STATUS "Git tag version: ${CPACK_PACKAGE_VERSION}")
    else()
        message(STATUS "No git tags found, using default version 1.0.0")
        set(CPACK_PACKAGE_VERSION "1.0.0")
    endif()
else()
    set(CPACK_PACKAGE_VERSION "1.0.0")
endif()