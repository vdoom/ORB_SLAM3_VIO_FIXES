#include <librealsense2/rs.hpp>
#include <iostream>

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
    
    auto profile = pipe.start(cfg);
    
    // Get the transformation from IMU to left camera
    auto accel_stream = profile.get_stream(RS2_STREAM_ACCEL).as<rs2::motion_stream_profile>();
    auto left_stream = profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
    
    auto accel_to_left = accel_stream.get_extrinsics_to(left_stream);
    
    std::cout << "IMU to Left Camera Transformation Matrix:" << std::endl;
    std::cout << "Rotation matrix:" << std::endl;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << accel_to_left.rotation[i*3 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Translation vector:" << std::endl;
    std::cout << accel_to_left.translation[0] << " " 
              << accel_to_left.translation[1] << " " 
              << accel_to_left.translation[2] << std::endl;
    
    // Format for YAML (4x4 homogeneous matrix)
    std::cout << "\nFor YAML config file (Tbc matrix):" << std::endl;
    std::cout << "Tbc: !!opencv-matrix" << std::endl;
    std::cout << "   rows: 4" << std::endl;
    std::cout << "   cols: 4" << std::endl;
    std::cout << "   dt: f" << std::endl;
    std::cout << "   data: [";
    
    // Print rotation matrix
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << accel_to_left.rotation[i*3 + j];
            if(i < 2 || j < 2) std::cout << ", ";
        }
        std::cout << ", " << accel_to_left.translation[i];
        if(i < 2) std::cout << "," << std::endl << "          ";
    }
    std::cout << "," << std::endl << "          0.0, 0.0, 0.0, 1.0]" << std::endl;
    
    return 0;
}
