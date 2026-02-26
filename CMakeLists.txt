cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)
 
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()
 
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
 
# Base flags with debug symbols
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")
 
# Release mode: O3 optimization + native arch
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
 
# RelWithDebInfo mode: O2 optimization + debug symbols (good for debugging)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -march=native")
 
# Debug mode: no optimization, full debug info
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
 
# Check C++14 support (required by Pangolin's sigslot)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()
 
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
 
find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()
 
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})
 
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)

# Optional CUDA acceleration for ORB extraction (Jetson Xavier NX, desktop NVIDIA GPU)
option(USE_CUDA "Enable CUDA acceleration for ORB extraction" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    # SM 7.2 = Xavier NX/AGX, SM 7.5 = Turing, SM 8.6 = Ampere desktop
    set(CMAKE_CUDA_ARCHITECTURES "72;75;86" CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    message(STATUS "CUDA acceleration ENABLED for ORB extraction")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
 
# Check for PearAPI (unified camera + IMU library)
# PearAPI is located in the parent VIO_Custom_Camera_Set directory
set(PEARAPI_DIR "${CMAKE_SOURCE_DIR}/../../PearAPI")
if(EXISTS "${PEARAPI_DIR}/CMakeLists.txt")
    message(STATUS "PearAPI found: ${PEARAPI_DIR}")
    add_subdirectory(${PEARAPI_DIR} ${CMAKE_BINARY_DIR}/PearAPI)
    set(PEARAPI_FOUND TRUE)
else()
    message(STATUS "PearAPI not found at ${PEARAPI_DIR} - custom camera examples will not be built")
    set(PEARAPI_FOUND FALSE)
endif()
 
# Include FetchContent for downloading dependencies
include(FetchContent)
 
# Fetch nlohmann/json library
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)
 
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)
 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
 
add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/VideoRecorder.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/VideoRecorder.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h)

# Add CUDA source files if enabled
if(USE_CUDA)
    target_sources(${PROJECT_NAME} PRIVATE
        src/cuda/ORBextractor_cuda.cu
        include/cuda/ORBextractor_cuda.h)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

add_subdirectory(Thirdparty/g2o)
 
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
)
 
# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()
 
##########################################################
# MAVLink
# Check for local MAVLink headers
set(MAVLINK_LOCAL_DIR "${CMAKE_SOURCE_DIR}/mavlink")
 
if(EXISTS "${MAVLINK_LOCAL_DIR}")
    message(STATUS "Using local MAVLink headers: ${MAVLINK_LOCAL_DIR}")
    set(MAVLINK_INCLUDE_DIR "${MAVLINK_LOCAL_DIR}")
else()
    # Try to download MAVLink
    message(STATUS "Downloading MAVLink headers...")
    include(FetchContent)
 
    FetchContent_Declare(
        mavlink
        GIT_REPOSITORY https://github.com/mavlink/c_library_v2.git
        GIT_TAG master
    )
 
    FetchContent_GetProperties(mavlink)
    if(NOT mavlink_POPULATED)
        FetchContent_Populate(mavlink)
        set(MAVLINK_INCLUDE_DIR "${mavlink_SOURCE_DIR}")
    endif()
endif()
 
# Verify headers exist
if(NOT EXISTS "${MAVLINK_INCLUDE_DIR}/common/mavlink.h")
    message(FATAL_ERROR "MAVLink headers not found! Run: ./download_mavlink.sh")
endif()
 
message(STATUS "MAVLink include: ${MAVLINK_INCLUDE_DIR}")
##########################################################
 
 
# Build examples
 
# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
 
add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()
 
if(realsense2_FOUND)
    add_executable(rgbd_realsense_d455
            Examples/RGB-D/rgbd_realsense_d455_simple.cc)
    target_link_libraries(rgbd_realsense_d455 ${PROJECT_NAME})
endif()
 
 
# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)
 
if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})

endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
 
add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})
 
add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})
 
add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})
 
    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()
 
#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
 
add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})
 
add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})
 
add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})
 
add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})
 
    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

# Custom camera with PearAPI (OV9281)
if(PEARAPI_FOUND)
    add_executable(mono_custom_cam
            Examples/Monocular/mono_custom_cam.cc)
    target_compile_features(mono_custom_cam PRIVATE cxx_std_17)
    target_link_libraries(mono_custom_cam ${PROJECT_NAME} PearAPI)
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)
 
add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})
 
add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})
 
    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()
 
# Custom camera with PearAPI (OV9281 + BMI160 via Pico)
if(PEARAPI_FOUND)
    add_executable(mono_inertial_custom_cam
            Examples/Monocular-Inertial/mono_inertial_custom_cam.cc)
    target_compile_features(mono_inertial_custom_cam PRIVATE cxx_std_17)
    target_link_libraries(mono_inertial_custom_cam ${PROJECT_NAME} PearAPI)

    # PearAPI + MAVLink VIOBridge (OV9281 + BMI160 + ArduPilot)
    add_executable(mono_inertial_pear
            Examples/Monocular-Inertial/mono_inertial_pear.cc)
    target_compile_features(mono_inertial_pear PRIVATE cxx_std_17)
    target_include_directories(mono_inertial_pear PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(mono_inertial_pear ${PROJECT_NAME} PearAPI Threads::Threads)

    # PearAPI + MAVLink GPS VIOBridge (OV9281 + BMI160 + ArduPilot GPS_INPUT)
    add_executable(mono_inertial_pear_fake_gps
            Examples/Monocular-Inertial/mono_inertial_pear_fake_gps.cc)
    target_compile_features(mono_inertial_pear_fake_gps PRIVATE cxx_std_17)
    target_include_directories(mono_inertial_pear_fake_gps PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(mono_inertial_pear_fake_gps ${PROJECT_NAME} PearAPI Threads::Threads)

    # PearAPI test/visualization only (OV9281 + BMI160, no MAVLink)
    add_executable(mono_inertial_pear_test
            Examples/Monocular-Inertial/mono_inertial_pear_test.cc)
    target_compile_features(mono_inertial_pear_test PRIVATE cxx_std_17)
    target_link_libraries(mono_inertial_pear_test ${PROJECT_NAME} PearAPI)
endif()
 
#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)
 
add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})
 
add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})
 
    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
 
    add_executable(stereo_inertial_realsense_D455
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455.cc)
    target_link_libraries(stereo_inertial_realsense_D455 ${PROJECT_NAME})
 
    add_executable(stereo_inertial_realsense_D455_2
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_2.cc)
    target_link_libraries(stereo_inertial_realsense_D455_2 ${PROJECT_NAME})
 
# Create library
add_library(uart_transfer_lib STATIC
    Examples/Stereo-Inertial/UARTTransfer.cpp
    Examples/Stereo-Inertial/UARTTransfer.h
)
target_link_libraries(uart_transfer_lib PRIVATE Threads::Threads)
 
    add_executable(stereo_inertial_realsense_D455_VIO
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_VIO.cc)
	target_include_directories(stereo_inertial_realsense_D455_VIO PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(stereo_inertial_realsense_D455_VIO ${PROJECT_NAME} nlohmann_json::nlohmann_json uart_transfer_lib Threads::Threads)

    # v2: Refactored VIO with coordinate frame bridging for ArduPilot integration
    add_executable(stereo_inertial_realsense_D455_VIO_v2
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_VIO_v2.cc)
    target_include_directories(stereo_inertial_realsense_D455_VIO_v2 PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(stereo_inertial_realsense_D455_VIO_v2 ${PROJECT_NAME} Threads::Threads)

    add_executable(stereo_inertial_realsense_D455_VIO_GPS
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_VIO_GPS.cc)
	target_include_directories(stereo_inertial_realsense_D455_VIO_GPS PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(stereo_inertial_realsense_D455_VIO_GPS ${PROJECT_NAME} nlohmann_json::nlohmann_json uart_transfer_lib Threads::Threads)
 
    add_executable(stereo_inertial_realsense_D455_VIO_GPS_v2
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_VIO_GPS_v2.cc)
	target_include_directories(stereo_inertial_realsense_D455_VIO_GPS_v2 PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(stereo_inertial_realsense_D455_VIO_GPS_v2 ${PROJECT_NAME} nlohmann_json::nlohmann_json uart_transfer_lib Threads::Threads)

    # v4: GPS VIO applying VIO v2 coordinate bridging approach to GPS output
    # Key improvements: VIOBridgeState, EKF feedback, map change detection via position jump,
    # STOP GPS messages when LOST (not send zeros), RECENTLY_LOST sends degraded quality
    add_executable(stereo_inertial_realsense_D455_VIO_GPS_v4
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_VIO_GPS_v4.cc)
    target_include_directories(stereo_inertial_realsense_D455_VIO_GPS_v4 PRIVATE ${MAVLINK_INCLUDE_DIR})
    target_link_libraries(stereo_inertial_realsense_D455_VIO_GPS_v4 ${PROJECT_NAME} Threads::Threads)

    add_executable(stereo_inertial_realsense_D455_axis_test
            Examples/Stereo-Inertial/stereo_inertial_realsense_D455_axis_test.cc)
    target_link_libraries(stereo_inertial_realsense_D455_axis_test ${PROJECT_NAME})
 
    add_executable(calibrate_d455
            Examples/Stereo-Inertial/calibrate_d455.cc)
    target_link_libraries(calibrate_d455 ${PROJECT_NAME})
 
    add_executable(calibrate_d455_imu
            Examples/Stereo-Inertial/calibrate_d455_imu.cc)
    target_link_libraries(calibrate_d455_imu ${PROJECT_NAME})
endif()
 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})
 
    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()
 
#Old examples
 
# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)
 
add_executable(rgbd_tum_old
        Examples_old/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            Examples_old/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
endif()
 
 
# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)
 
if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()
 
#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)
 
add_executable(stereo_kitti_old
        Examples_old/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${PROJECT_NAME})
 
add_executable(stereo_euroc_old
        Examples_old/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${PROJECT_NAME})
 
add_executable(stereo_tum_vi_old
        Examples_old/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Examples_old/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})
 
    add_executable(stereo_realsense_D435i_old
            Examples_old/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
endif()
 
#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)
 
add_executable(mono_tum_old
        Examples_old/Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${PROJECT_NAME})
 
add_executable(mono_kitti_old
        Examples_old/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${PROJECT_NAME})
 
add_executable(mono_euroc_old
        Examples_old/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${PROJECT_NAME})
 
add_executable(mono_tum_vi_old
        Examples_old/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Examples_old/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})
 
    add_executable(mono_realsense_D435i_old
            Examples_old/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
endif()
 
#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)
 
add_executable(mono_inertial_euroc_old
        Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})
 
add_executable(mono_inertial_tum_vi_old
        Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})
 
    add_executable(mono_inertial_realsense_D435i_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()
 
#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)
 
add_executable(stereo_inertial_euroc_old
        Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})
 
add_executable(stereo_inertial_tum_vi_old
        Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})
 
if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})
 
    add_executable(stereo_inertial_realsense_D435i_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()
 