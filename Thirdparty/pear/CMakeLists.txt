cmake_minimum_required(VERSION 3.10)
project(pear VERSION 1.0.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Output directories - keep everything local
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Find dependencies
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# Source files
set(PEAR_SOURCES
    src/camera.cpp
    src/imu.cpp
    src/pear_vio_camera.cpp
)

# Header files (for IDE support)
set(PEAR_HEADERS
    include/pear/pear.h
    include/pear/types.h
    include/pear/config.h
    include/pear/camera.h
    include/pear/imu.h
    include/pear/pear_vio_camera.h
)

# Shared library
add_library(pear SHARED ${PEAR_SOURCES} ${PEAR_HEADERS})
target_include_directories(pear
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LIBCAMERA_INCLUDE_DIRS}
)
target_compile_options(pear PRIVATE ${LIBCAMERA_CFLAGS_OTHER})
target_link_libraries(pear
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        ${LIBCAMERA_LIBRARIES}
)
set_target_properties(pear PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME pear
)

# Static library
add_library(pear_static STATIC ${PEAR_SOURCES} ${PEAR_HEADERS})
target_include_directories(pear_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LIBCAMERA_INCLUDE_DIRS}
)
target_compile_options(pear_static PRIVATE ${LIBCAMERA_CFLAGS_OTHER})
target_link_libraries(pear_static
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        ${LIBCAMERA_LIBRARIES}
)
set_target_properties(pear_static PROPERTIES
    OUTPUT_NAME pear
)

# Print configuration
message(STATUS "Pear VIO Camera Library")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  libcamera: ${LIBCAMERA_VERSION}")
message(STATUS "  Output: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
