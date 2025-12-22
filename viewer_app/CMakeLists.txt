cmake_minimum_required(VERSION 3.16)
project(rpi_camera_viewer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Widgets Gui)

# Find libcamera (same as rpicam-hello)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# Source files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/CameraWidget.cpp
    src/CameraController.cpp
    src/Axes3DWidget.cpp
    src/SerialIMUReader.cpp
    src/RosBagWriter.cpp
    src/RosbagRecorder.cpp
)

set(HEADERS
    src/MainWindow.h
    src/CameraWidget.h
    src/CameraController.h
    src/Axes3DWidget.h
    src/SerialIMUReader.h
    src/RosBagWriter.h
    src/RosbagRecorder.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBCAMERA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Widgets
    Qt6::Gui
    ${LIBCAMERA_LIBRARIES}
)

target_compile_options(${PROJECT_NAME} PRIVATE ${LIBCAMERA_CFLAGS_OTHER})

# Install
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
