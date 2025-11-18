#include <librealsense2/rs.hpp>
#include <iostream>

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    
    auto profile = pipe.start(cfg);
    
    // Get intrinsics for left camera
    auto left_stream = profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
    auto left_intrinsics = left_stream.get_intrinsics();
    
    // Get intrinsics for right camera  
    auto right_stream = profile.get_stream(RS2_STREAM_INFRARED, 2).as<rs2::video_stream_profile>();
    auto right_intrinsics = right_stream.get_intrinsics();
    
    // Get extrinsics (stereo baseline)
    auto left_to_right = left_stream.get_extrinsics_to(right_stream);
    
    std::cout << "Left Camera:" << std::endl;
    std::cout << "fx: " << left_intrinsics.fx << ", fy: " << left_intrinsics.fy << std::endl;
    std::cout << "cx: " << left_intrinsics.ppx << ", cy: " << left_intrinsics.ppy << std::endl;
    std::cout << "Distortion: " << left_intrinsics.coeffs[0] << ", " << left_intrinsics.coeffs[1] << ", " 
              << left_intrinsics.coeffs[2] << ", " << left_intrinsics.coeffs[3] << ", " << left_intrinsics.coeffs[4] << std::endl;
    
    std::cout << "\nRight Camera:" << std::endl;
    std::cout << "fx: " << right_intrinsics.fx << ", fy: " << right_intrinsics.fy << std::endl;
    std::cout << "cx: " << right_intrinsics.ppx << ", cy: " << right_intrinsics.ppy << std::endl;
    
    std::cout << "\nBaseline: " << sqrt(left_to_right.translation[0]*left_to_right.translation[0] + 
                                       left_to_right.translation[1]*left_to_right.translation[1] + 
                                       left_to_right.translation[2]*left_to_right.translation[2]) << " meters" << std::endl;
    
    return 0;
}
